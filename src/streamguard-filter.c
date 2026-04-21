/*
StreamGuard filter — video filter that buffers frames, runs Apple Vision
OCR on the freshest frame each tick, detects secrets, and outputs delayed
frames with censor mask applied.

The frame buffer (ring of N+1 texrenders) is the key trick: we output a
frame that was sampled ~30 ticks ago, so by the time it reaches the
encoder, OCR has already inspected it and our region list reflects its
contents. Without the buffer, a newly appearing secret would be visible
on stream for 100–600ms until OCR caught up — unsafe for the threat
model.
*/

#include <obs-module.h>
#include <plugin-support.h>
#include <util/platform.h>
#include <util/threading.h>
#include <pthread.h>
#include <string.h>

#include "vision-ocr.h"
#include "secret-detector.h"

#define OCR_INTERVAL_NS_DEFAULT 250000000ULL    /* 250 ms = 4 Hz */
#define REGION_TTL_NS_DEFAULT   2500000000ULL   /* keep censor up 2.5s after last detect */
#define REGION_PAD_PCT_DEFAULT  0.30f           /* baseline padding on every side */
/* Age-based inflation: by the time a region hits this age (≈ OCR period)
 * we've added AGE_INFLATION_MAX on every side on top of the base padding.
 * Covers motion within the inter-tick dead time, and also compensates for
 * OCR sometimes missing parts of a string that a neighbour tick caught. */
#define REGION_AGE_INFLATION_MAX 0.25f
#define REGION_AGE_INFLATION_FULL_NS 300000000ULL
#define MAX_REGIONS             64
#define MASK_W                  320
#define MASK_H                  180

/* Ring size = delay + 2. The +2 is load-bearing: we write first, then read
 * in the same tick. With cap = delay + 1, the read slot and the just-written
 * slot are identical after the first wrap, so "delayed" output = freshly
 * captured pixels and the delay collapses to zero. The extra slot keeps
 * write and read always pointing at different textures. */
#define BUFFER_DELAY_FRAMES     45              /* ~750ms at 60fps */
#define BUFFER_CAP              (BUFFER_DELAY_FRAMES + 2)

struct sg_region {
	float x, y, w, h; /* normalized UV, top-left origin */
	uint64_t last_seen_ns;
};

struct streamguard_filter {
	obs_source_t *source;

	/* Delay ring: each slot holds one captured source frame. */
	gs_texrender_t *ring[BUFFER_CAP];
	uint32_t ring_w;
	uint32_t ring_h;
	int write_idx;
	int read_idx;
	int items;

	gs_stagesurf_t *stagesurface;
	uint32_t stage_w;
	uint32_t stage_h;

	sg_ocr_ctx *ocr;
	sg_detector *detector;
	uint64_t ocr_interval_ns;
	uint64_t last_ocr_ns;
	uint64_t next_frame_id;

	gs_effect_t *censor_effect;
	gs_eparam_t *param_image;
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

	for (int i = 0; i < f->region_count; i++) {
		struct sg_region *r = &f->regions[i];
		float ix = fmaxf(r->x, x);
		float iy = fmaxf(r->y, y);
		float ax = fminf(r->x + r->w, x + w);
		float ay = fminf(r->y + r->h, y + h);
		if (ax > ix && ay > iy) {
			/* Union with existing rect. OCR bboxes fluctuate between
			 * ticks — one tick may only catch "AKIA..." while the next
			 * catches the whole key. Replacing would briefly expose the
			 * missed letters; unioning keeps the widest envelope we've
			 * ever seen for this text. */
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
		r->x = x; r->y = y; r->w = w; r->h = h;
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

	if (f && f->detector && result->count > 0) {
		bool *flags = bzalloc(sizeof(bool) * (size_t)result->count);
		const char **rules =
			bzalloc(sizeof(const char *) * (size_t)result->count);
		sg_detector_check_all(f->detector, result->boxes, result->count, flags,
				      rules);

		int hits = 0;
		for (int i = 0; i < result->count; i++) {
			if (!flags[i])
				continue;
			hits++;
			sg_ocr_box *b = &result->boxes[i];
			/* Vision bottom-left → UV top-left */
			float uv_x = b->x;
			float uv_y = 1.0f - b->y - b->h;
			streamguard_add_region(f, uv_x, uv_y, b->w, b->h, now);
			obs_log(LOG_WARNING,
				"SECRET frame=%llu rule=%s bbox=(%.2f,%.2f %.2fx%.2f) text=\"%s\"",
				(unsigned long long)result->frame_id,
				rules[i] ? rules[i] : "?", b->x, b->y, b->w, b->h, b->text);
		}
		if (hits > 0) {
			obs_log(LOG_WARNING, "frame %llu: %d/%d boxes flagged",
				(unsigned long long)result->frame_id, hits, result->count);
		}

		bfree(flags);
		bfree(rules);
	}

	sg_ocr_free_result(result);
}

static void streamguard_build_mask(struct streamguard_filter *f, uint64_t now_ns)
{
	memset(f->mask_buf, 0, sizeof(f->mask_buf));
	pthread_mutex_lock(&f->regions_mutex);
	for (int i = 0; i < f->region_count; i++) {
		struct sg_region *r = &f->regions[i];

		/* Age-based inflation: a region that was freshly detected this
		 * tick draws at its base size; one getting close to the next
		 * OCR tick draws inflated to catch motion happening in the
		 * dead time. Yes this causes visible pulsing at the OCR rate
		 * on static content — user-preferred trade vs motion leaks. */
		uint64_t age_ns = now_ns - r->last_seen_ns;
		float age_factor = (float)age_ns / (float)REGION_AGE_INFLATION_FULL_NS;
		if (age_factor > 1.0f) age_factor = 1.0f;
		float inflate = age_factor * REGION_AGE_INFLATION_MAX;

		float pad_w = r->w * inflate;
		float pad_h = r->h * inflate;
		float rx = r->x - pad_w;
		float ry = r->y - pad_h;
		float rw = r->w + 2.0f * pad_w;
		float rh = r->h + 2.0f * pad_h;

		int x0 = (int)(rx * MASK_W);
		int y0 = (int)(ry * MASK_H);
		int x1 = (int)((rx + rw) * MASK_W + 0.5f);
		int y1 = (int)((ry + rh) * MASK_H + 0.5f);
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
	f->region_ttl_ns = REGION_TTL_NS_DEFAULT;
	f->region_pad_pct = REGION_PAD_PCT_DEFAULT;
	pthread_mutex_init(&f->regions_mutex, NULL);

	obs_enter_graphics();
	for (int i = 0; i < BUFFER_CAP; i++) {
		f->ring[i] = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	}
	char *effect_path = obs_module_file("streamguard-censor.effect");
	if (effect_path) {
		f->censor_effect = gs_effect_create_from_file(effect_path, NULL);
		bfree(effect_path);
	}
	if (f->censor_effect) {
		f->param_image = gs_effect_get_param_by_name(f->censor_effect, "image");
		f->param_mask = gs_effect_get_param_by_name(f->censor_effect, "mask");
		obs_log(LOG_INFO, "censor effect loaded: image=%p mask=%p",
			(void *)f->param_image, (void *)f->param_mask);
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

	if (f->ocr) { sg_ocr_destroy(f->ocr); f->ocr = NULL; }
	if (f->detector) { sg_detector_destroy(f->detector); f->detector = NULL; }

	obs_enter_graphics();
	for (int i = 0; i < BUFFER_CAP; i++) {
		if (f->ring[i]) gs_texrender_destroy(f->ring[i]);
	}
	if (f->stagesurface) gs_stagesurface_destroy(f->stagesurface);
	if (f->mask_tex) gs_texture_destroy(f->mask_tex);
	if (f->censor_effect) gs_effect_destroy(f->censor_effect);
	obs_leave_graphics();

	pthread_mutex_destroy(&f->regions_mutex);
	bfree(f);
}

/* Sample the source into ring[write_idx]. Returns the texture on success. */
static gs_texture_t *streamguard_capture_source(struct streamguard_filter *f, uint32_t w,
						uint32_t h)
{
	gs_texrender_t *tr = f->ring[f->write_idx];
	gs_texrender_reset(tr);
	if (!gs_texrender_begin(tr, w, h))
		return NULL;

	struct vec4 bg;
	vec4_zero(&bg);
	gs_clear(GS_CLEAR_COLOR, &bg, 0.0f, 0);
	gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);

	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_t *target = obs_filter_get_target(f->source);
	if (target)
		obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(tr);

	return gs_texrender_get_texture(tr);
}

static void streamguard_submit_ocr(struct streamguard_filter *f, gs_texture_t *source_tex,
				   uint32_t w, uint32_t h)
{
	if (!source_tex)
		return;

	if (f->stagesurface && (f->stage_w != w || f->stage_h != h)) {
		gs_stagesurface_destroy(f->stagesurface);
		f->stagesurface = NULL;
	}
	if (!f->stagesurface) {
		f->stagesurface = gs_stagesurface_create(w, h, GS_BGRA);
		f->stage_w = w;
		f->stage_h = h;
	}

	gs_stage_texture(f->stagesurface, source_tex);

	uint8_t *video_data = NULL;
	uint32_t linesize = 0;
	if (!gs_stagesurface_map(f->stagesurface, &video_data, &linesize))
		return;

	sg_ocr_submit(f->ocr, video_data, (int)w, (int)h, (int)linesize, f->next_frame_id++,
		      streamguard_ocr_done, f);
	gs_stagesurface_unmap(f->stagesurface);
}

static void streamguard_draw_black(uint32_t w, uint32_t h)
{
	gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
	if (!solid)
		return;
	gs_eparam_t *color = gs_effect_get_param_by_name(solid, "color");
	struct vec4 black = {0.0f, 0.0f, 0.0f, 1.0f};
	gs_effect_set_vec4(color, &black);
	while (gs_effect_loop(solid, "Solid")) {
		gs_draw_sprite(NULL, 0, w, h);
	}
}

static void streamguard_filter_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct streamguard_filter *f = data;

	obs_source_t *target = obs_filter_get_target(f->source);
	uint32_t w = target ? obs_source_get_base_width(target) : 0;
	uint32_t h = target ? obs_source_get_base_height(target) : 0;
	if (!target || w == 0 || h == 0 || !f->censor_effect) {
		obs_source_skip_video_filter(f->source);
		return;
	}

	/* 1. Capture current source frame into ring[write_idx]. */
	gs_texture_t *fresh = streamguard_capture_source(f, w, h);
	if (!fresh) {
		obs_source_skip_video_filter(f->source);
		return;
	}

	/* 2. Kick OCR on the fresh frame at the throttled rate. OCR submit
	 * makes its own copy of the pixel data, so the ring slot is free to
	 * be overwritten on the next tick. */
	uint64_t now = os_gettime_ns();
	if (now - f->last_ocr_ns >= f->ocr_interval_ns) {
		streamguard_submit_ocr(f, fresh, w, h);
		f->last_ocr_ns = now;
	}

	/* 3. Expire regions older than TTL. */
	streamguard_prune_regions(f, now);

	/* 4. Rebuild the censor mask from current regions (with motion extrapolation). */
	streamguard_build_mask(f, now);
	gs_texture_set_image(f->mask_tex, f->mask_buf, MASK_W, false);

	/* Once a second: visibility into the pipeline state. */
	static uint64_t s_last_diag_ns = 0;
	if (now - s_last_diag_ns > 1000000000ULL) {
		s_last_diag_ns = now;
		pthread_mutex_lock(&f->regions_mutex);
		int rc = f->region_count;
		pthread_mutex_unlock(&f->regions_mutex);
		obs_log(LOG_INFO,
			"render: items=%d/%d regions=%d write=%d read=%d", f->items,
			BUFFER_DELAY_FRAMES, rc, f->write_idx, f->read_idx);
	}

	/* 5. Output: delayed frame through censor effect, or black during warmup. */
	if (f->items > BUFFER_DELAY_FRAMES) {
		gs_texture_t *delayed = gs_texrender_get_texture(f->ring[f->read_idx]);
		if (delayed) {
			gs_effect_set_texture(f->param_image, delayed);
			gs_effect_set_texture(f->param_mask, f->mask_tex);
			while (gs_effect_loop(f->censor_effect, "Draw")) {
				gs_draw_sprite(delayed, 0, w, h);
			}
		}
		f->read_idx = (f->read_idx + 1) % BUFFER_CAP;
		f->items--;
	} else {
		streamguard_draw_black(w, h);
	}

	/* 6. Advance the writer. */
	f->write_idx = (f->write_idx + 1) % BUFFER_CAP;
	f->items++;
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
