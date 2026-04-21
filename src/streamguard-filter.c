/*
StreamGuard filter — video filter that periodically reads back frames from
the GPU, runs Apple Vision OCR on them, detects secrets, and censors the
detected regions with solid black rectangles via streamguard-censor.effect.
*/

#include <obs-module.h>
#include <plugin-support.h>
#include <util/platform.h>
#include <util/threading.h>
#include <pthread.h>
#include <string.h>

#include "vision-ocr.h"
#include "secret-detector.h"

#define OCR_INTERVAL_NS_DEFAULT 500000000ULL    /* 500 ms = 2 Hz */
#define REGION_TTL_NS_DEFAULT   1500000000ULL   /* keep censor up 1.5s after last detect */
#define REGION_PAD_PCT_DEFAULT  0.10f           /* extend each detection by 10% of its size */
#define MAX_REGIONS             64
#define MASK_W                  320
#define MASK_H                  180

struct sg_region {
	float x, y, w, h; /* normalized UV, top-left origin */
	uint64_t last_seen_ns;
};

struct streamguard_filter {
	obs_source_t *source;

	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;
	uint32_t stage_w;
	uint32_t stage_h;

	sg_ocr_ctx *ocr;
	sg_detector *detector;
	uint64_t ocr_interval_ns;
	uint64_t last_ocr_ns;
	uint64_t next_frame_id;

	gs_effect_t *censor_effect;
	gs_eparam_t *param_mask;

	gs_texture_t *mask_tex;
	uint8_t mask_buf[MASK_W * MASK_H];

	pthread_mutex_t regions_mutex;
	struct sg_region regions[MAX_REGIONS];
	int region_count;
	uint64_t region_ttl_ns;
	float region_pad_pct;
};

static const char *streamguard_filter_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("StreamGuard.Filter");
}

static void streamguard_add_region(struct streamguard_filter *f, float x, float y, float w, float h,
				   uint64_t now_ns)
{
	/* Pad the box so letters at the edges don't peek out of the censor. */
	float pad_w = w * f->region_pad_pct;
	float pad_h = h * f->region_pad_pct;
	x -= pad_w;
	y -= pad_h;
	w += 2.0f * pad_w;
	h += 2.0f * pad_h;
	if (x < 0.0f) { w += x; x = 0.0f; }
	if (y < 0.0f) { h += y; y = 0.0f; }
	if (x + w > 1.0f) w = 1.0f - x;
	if (y + h > 1.0f) h = 1.0f - y;
	if (w <= 0.0f || h <= 0.0f)
		return;

	pthread_mutex_lock(&f->regions_mutex);

	/* Merge if the new rect substantially overlaps an existing one —
	 * otherwise the array fills quickly with near-duplicates when OCR
	 * re-detects the same text frame after frame. */
	for (int i = 0; i < f->region_count; i++) {
		struct sg_region *r = &f->regions[i];
		float ix = fmaxf(r->x, x);
		float iy = fmaxf(r->y, y);
		float ax = fminf(r->x + r->w, x + w);
		float ay = fminf(r->y + r->h, y + h);
		if (ax > ix && ay > iy) {
			/* Expand to union so neither detection escapes the censor. */
			float nx = fminf(r->x, x);
			float ny = fminf(r->y, y);
			r->w = fmaxf(r->x + r->w, x + w) - nx;
			r->h = fmaxf(r->y + r->h, y + h) - ny;
			r->x = nx;
			r->y = ny;
			r->last_seen_ns = now_ns;
			pthread_mutex_unlock(&f->regions_mutex);
			return;
		}
	}

	if (f->region_count < MAX_REGIONS) {
		struct sg_region *r = &f->regions[f->region_count++];
		r->x = x;
		r->y = y;
		r->w = w;
		r->h = h;
		r->last_seen_ns = now_ns;
	}
	pthread_mutex_unlock(&f->regions_mutex);
}

static void streamguard_prune_regions(struct streamguard_filter *f, uint64_t now_ns)
{
	pthread_mutex_lock(&f->regions_mutex);
	int w = 0;
	for (int i = 0; i < f->region_count; i++) {
		if (now_ns - f->regions[i].last_seen_ns < f->region_ttl_ns) {
			f->regions[w++] = f->regions[i];
		}
	}
	f->region_count = w;
	pthread_mutex_unlock(&f->regions_mutex);
}

static void streamguard_ocr_done(sg_ocr_result *result, void *user_data)
{
	struct streamguard_filter *f = user_data;
	if (!result)
		return;

	uint64_t now = os_gettime_ns();
	int hits = 0;
	for (int i = 0; i < result->count; i++) {
		sg_ocr_box *b = &result->boxes[i];
		const char *rule = NULL;
		if (f && f->detector && sg_detector_check(f->detector, b->text, &rule)) {
			hits++;
			/* Vision: bottom-left origin. UV: top-left. Flip Y. */
			float uv_x = b->x;
			float uv_y = 1.0f - b->y - b->h;
			streamguard_add_region(f, uv_x, uv_y, b->w, b->h, now);
			obs_log(LOG_WARNING,
				"SECRET frame=%llu rule=%s bbox=(%.2f,%.2f %.2fx%.2f) text=\"%s\"",
				(unsigned long long)result->frame_id, rule ? rule : "?", b->x,
				b->y, b->w, b->h, b->text);
		}
	}
	if (hits > 0) {
		obs_log(LOG_WARNING, "frame %llu: %d/%d boxes flagged",
			(unsigned long long)result->frame_id, hits, result->count);
	}
	sg_ocr_free_result(result);
}

static void streamguard_build_mask(struct streamguard_filter *f)
{
	memset(f->mask_buf, 0, sizeof(f->mask_buf));

	pthread_mutex_lock(&f->regions_mutex);
	for (int i = 0; i < f->region_count; i++) {
		struct sg_region *r = &f->regions[i];
		int x0 = (int)(r->x * MASK_W);
		int y0 = (int)(r->y * MASK_H);
		int x1 = (int)((r->x + r->w) * MASK_W + 0.5f);
		int y1 = (int)((r->y + r->h) * MASK_H + 0.5f);
		if (x0 < 0) x0 = 0;
		if (y0 < 0) y0 = 0;
		if (x1 > MASK_W) x1 = MASK_W;
		if (y1 > MASK_H) y1 = MASK_H;
		for (int y = y0; y < y1; y++) {
			memset(&f->mask_buf[y * MASK_W + x0], 0xFF, (size_t)(x1 - x0));
		}
	}
	pthread_mutex_unlock(&f->regions_mutex);
}

static void *streamguard_filter_create(obs_data_t *settings, obs_source_t *source)
{
	UNUSED_PARAMETER(settings);
	struct streamguard_filter *f = bzalloc(sizeof(struct streamguard_filter));
	f->source = source;
	f->ocr_interval_ns = OCR_INTERVAL_NS_DEFAULT;
	f->last_ocr_ns = 0;
	f->next_frame_id = 0;
	f->region_ttl_ns = REGION_TTL_NS_DEFAULT;
	f->region_pad_pct = REGION_PAD_PCT_DEFAULT;
	pthread_mutex_init(&f->regions_mutex, NULL);

	obs_enter_graphics();
	f->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	char *effect_path = obs_module_file("streamguard-censor.effect");
	if (effect_path) {
		f->censor_effect = gs_effect_create_from_file(effect_path, NULL);
		bfree(effect_path);
	}
	if (f->censor_effect) {
		f->param_mask = gs_effect_get_param_by_name(f->censor_effect, "mask");
		obs_log(LOG_INFO, "censor effect loaded: mask_param=%p",
			(void *)f->param_mask);
	} else {
		obs_log(LOG_ERROR, "failed to load streamguard-censor.effect");
	}
	const uint8_t *ptrs[1] = {f->mask_buf};
	f->mask_tex = gs_texture_create(MASK_W, MASK_H, GS_R8, 1, ptrs, GS_DYNAMIC);
	obs_leave_graphics();

	f->ocr = sg_ocr_create();
	f->detector = sg_detector_create();
	return f;
}

static void streamguard_filter_destroy(void *data)
{
	struct streamguard_filter *f = data;
	if (!f)
		return;

	if (f->ocr) {
		sg_ocr_destroy(f->ocr);
		f->ocr = NULL;
	}
	if (f->detector) {
		sg_detector_destroy(f->detector);
		f->detector = NULL;
	}

	obs_enter_graphics();
	if (f->stagesurface) {
		gs_stagesurface_destroy(f->stagesurface);
		f->stagesurface = NULL;
	}
	if (f->texrender) {
		gs_texrender_destroy(f->texrender);
		f->texrender = NULL;
	}
	if (f->mask_tex) {
		gs_texture_destroy(f->mask_tex);
		f->mask_tex = NULL;
	}
	if (f->censor_effect) {
		gs_effect_destroy(f->censor_effect);
		f->censor_effect = NULL;
	}
	obs_leave_graphics();

	pthread_mutex_destroy(&f->regions_mutex);
	bfree(f);
}

static bool streamguard_readback_and_submit(struct streamguard_filter *f)
{
	obs_source_t *target = obs_filter_get_target(f->source);
	if (!target)
		return false;

	uint32_t width = obs_source_get_base_width(target);
	uint32_t height = obs_source_get_base_height(target);
	if (width == 0 || height == 0)
		return false;

	gs_texrender_reset(f->texrender);
	if (!gs_texrender_begin(f->texrender, width, height))
		return false;

	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, (float)width, 0.0f, (float)height, -100.0f, 100.0f);

	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(f->texrender);

	if (f->stagesurface && (f->stage_w != width || f->stage_h != height)) {
		gs_stagesurface_destroy(f->stagesurface);
		f->stagesurface = NULL;
	}
	if (!f->stagesurface) {
		f->stagesurface = gs_stagesurface_create(width, height, GS_BGRA);
		f->stage_w = width;
		f->stage_h = height;
	}

	gs_stage_texture(f->stagesurface, gs_texrender_get_texture(f->texrender));

	uint8_t *video_data = NULL;
	uint32_t linesize = 0;
	if (!gs_stagesurface_map(f->stagesurface, &video_data, &linesize))
		return false;

	bool submitted = sg_ocr_submit(f->ocr, video_data, (int)width, (int)height,
				       (int)linesize, f->next_frame_id++, streamguard_ocr_done,
				       f);
	gs_stagesurface_unmap(f->stagesurface);
	return submitted;
}

static void streamguard_filter_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct streamguard_filter *f = data;

	uint64_t now = os_gettime_ns();
	if (now - f->last_ocr_ns >= f->ocr_interval_ns) {
		if (streamguard_readback_and_submit(f)) {
			f->last_ocr_ns = now;
		}
	}

	streamguard_prune_regions(f, now);

	if (!f->censor_effect || !f->mask_tex) {
		obs_source_skip_video_filter(f->source);
		return;
	}

	streamguard_build_mask(f);
	gs_texture_set_image(f->mask_tex, f->mask_buf, MASK_W, false);

	/* Once a second: what does the render path see? */
	static uint64_t s_last_diag_ns = 0;
	if (now - s_last_diag_ns > 1000000000ULL) {
		s_last_diag_ns = now;
		pthread_mutex_lock(&f->regions_mutex);
		int rc = f->region_count;
		float rx = rc > 0 ? f->regions[0].x : 0.0f;
		float ry = rc > 0 ? f->regions[0].y : 0.0f;
		float rw = rc > 0 ? f->regions[0].w : 0.0f;
		float rh = rc > 0 ? f->regions[0].h : 0.0f;
		pthread_mutex_unlock(&f->regions_mutex);
		obs_log(LOG_INFO,
			"render: regions=%d first=(%.2f,%.2f %.2fx%.2f) effect=%p mask=%p",
			rc, rx, ry, rw, rh, (void *)f->censor_effect,
			(void *)f->mask_tex);
	}

	if (!obs_source_process_filter_begin(f->source, GS_RGBA, OBS_NO_DIRECT_RENDERING))
		return;

	if (f->param_mask)
		gs_effect_set_texture(f->param_mask, f->mask_tex);

	obs_source_process_filter_end(f->source, f->censor_effect,
				      obs_source_get_base_width(f->source),
				      obs_source_get_base_height(f->source));
}

struct obs_source_info streamguard_filter_info = {
	.id = "streamguard_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = streamguard_filter_get_name,
	.create = streamguard_filter_create,
	.destroy = streamguard_filter_destroy,
	.video_render = streamguard_filter_video_render,
};
