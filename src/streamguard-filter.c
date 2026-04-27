/*
StreamGuard filter — buffers frames, runs Apple Vision OCR on the freshest
frame each tick, detects secrets, and outputs delayed frames with a censor
mask applied. Most pipeline knobs are exposed in the filter properties:

  - "Max processing resolution"    → pipeline_max_dim
  - "OCR rate when active"         → ocr_active_ns
  - "OCR rate when idle"           → ocr_idle_ns
  - "Buffer delay (ms)"            → effective_delay_frames
  - "Blur strength"                → blur_radius
  - "Don't blur URLs"              → detector flag
  - "Detect random-looking …"      → detector flag (entropy)
  - "Detect by label proximity"    → detector flag

The frame buffer (ring of N+1 texrenders) is the safety mechanism: we
output a frame that was sampled `effective_delay_frames` ticks ago, so
by the time it reaches the encoder Vision has had time to inspect it
and our region list reflects its contents.
*/

#include <obs-module.h>
#include <plugin-support.h>
#include <util/platform.h>
#include <util/threading.h>
#include <pthread.h>
#include <string.h>

#include "vision-ocr.h"
#include "secret-detector.h"

/* The ring is sized for the largest delay we expose in the UI (1500ms at
 * 60 fps = 90 frames + 2). Slots that aren't used in the current setting
 * never get rendered into and so cost no GPU memory — gs_texrender only
 * allocates a backing texture on first gs_texrender_begin. */
#define MAX_DELAY_FRAMES        90
#define BUFFER_CAP              (MAX_DELAY_FRAMES + 2)

#define DEFAULT_DELAY_MS        500
#define DEFAULT_OCR_HZ          4
#define DEFAULT_OCR_IDLE_HZ     2
#define DEFAULT_MAX_DIM         1080
/* Switch to idle rate once we've gone this many OCR ticks with zero hits. */
#define IDLE_THRESHOLD_TICKS    8

#define REGION_TTL_NS_DEFAULT   2500000000ULL
#define REGION_PAD_PCT_DEFAULT  0.30f
#define REGION_AGE_INFLATION_MAX     0.25f
#define REGION_AGE_INFLATION_FULL_NS 300000000ULL
#define MAX_REGIONS             64
#define MASK_W                  320
#define MASK_H                  180

/* "Blur strength" enum values, mapped to UV radius. */
#define BLUR_LOW     0.008f
#define BLUR_MEDIUM  0.015f
#define BLUR_HIGH    0.025f

struct sg_region {
	float x, y, w, h;
	uint64_t last_seen_ns;
};

struct streamguard_filter {
	obs_source_t *source;

	/* Delay ring. */
	gs_texrender_t *ring[BUFFER_CAP];
	int write_idx;
	int read_idx;
	int items;
	int effective_delay_frames; /* updated from the delay-ms setting */
	int effective_cap;          /* effective_delay_frames + 2 */

	gs_stagesurf_t *stagesurface;
	uint32_t stage_w;
	uint32_t stage_h;

	sg_ocr_ctx *ocr;
	sg_detector *detector;

	/* OCR pacing — adaptive between active and idle rate. */
	uint64_t ocr_active_ns;
	uint64_t ocr_idle_ns;
	uint64_t last_ocr_ns;
	uint64_t next_frame_id;
	int idle_streak; /* OCR ticks in a row with zero hits */

	/* Pipeline cap: render-time downsample target. 0 = no cap. */
	int pipeline_max_dim;

	gs_effect_t *censor_effect;
	gs_eparam_t *param_image;
	gs_eparam_t *param_mask;
	gs_eparam_t *param_blur_radius;
	float blur_radius;

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
	int hits = 0;

	if (f && f->detector && result->count > 0) {
		bool *flags = bzalloc(sizeof(bool) * (size_t)result->count);
		const char **rules =
			bzalloc(sizeof(const char *) * (size_t)result->count);
		sg_detector_check_all(f->detector, result->boxes, result->count, flags,
				      rules);

		for (int i = 0; i < result->count; i++) {
			if (!flags[i])
				continue;
			hits++;
			sg_ocr_box *b = &result->boxes[i];
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

	if (f) {
		if (hits > 0)
			f->idle_streak = 0;
		else
			f->idle_streak++;
	}

	sg_ocr_free_result(result);
}

static void streamguard_build_mask(struct streamguard_filter *f, uint64_t now_ns)
{
	memset(f->mask_buf, 0, sizeof(f->mask_buf));
	pthread_mutex_lock(&f->regions_mutex);
	for (int i = 0; i < f->region_count; i++) {
		struct sg_region *r = &f->regions[i];
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

static obs_properties_t *streamguard_filter_get_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();
	obs_property_t *p;

	p = obs_properties_add_list(props, "max_resolution",
				    obs_module_text("StreamGuard.MaxResolution"),
				    OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p, obs_module_text("StreamGuard.MaxResolution.Native"), 0);
	obs_property_list_add_int(p, obs_module_text("StreamGuard.MaxResolution.720"), 720);
	obs_property_list_add_int(p, obs_module_text("StreamGuard.MaxResolution.1080"), 1080);
	obs_property_list_add_int(p, obs_module_text("StreamGuard.MaxResolution.1440"), 1440);
	obs_property_set_long_description(
		p, obs_module_text("StreamGuard.MaxResolution.Description"));

	p = obs_properties_add_int_slider(props, "ocr_active_hz",
					  obs_module_text("StreamGuard.OcrRateHz"), 1, 10, 1);
	obs_property_set_long_description(p,
					  obs_module_text("StreamGuard.OcrRateHz.Description"));

	p = obs_properties_add_int_slider(props, "ocr_idle_hz",
					  obs_module_text("StreamGuard.OcrIdleRateHz"), 1, 6, 1);
	obs_property_set_long_description(
		p, obs_module_text("StreamGuard.OcrIdleRateHz.Description"));

	p = obs_properties_add_int_slider(props, "buffer_delay_ms",
					  obs_module_text("StreamGuard.BufferDelayMs"), 300,
					  1500, 50);
	obs_property_set_long_description(
		p, obs_module_text("StreamGuard.BufferDelayMs.Description"));

	p = obs_properties_add_list(props, "blur_strength",
				    obs_module_text("StreamGuard.BlurStrength"),
				    OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p, obs_module_text("StreamGuard.BlurStrength.Low"), 0);
	obs_property_list_add_int(p, obs_module_text("StreamGuard.BlurStrength.Medium"), 1);
	obs_property_list_add_int(p, obs_module_text("StreamGuard.BlurStrength.High"), 2);
	obs_property_set_long_description(p,
					  obs_module_text("StreamGuard.BlurStrength.Description"));

	p = obs_properties_add_bool(props, "ignore_urls",
				    obs_module_text("StreamGuard.IgnoreUrls"));
	obs_property_set_long_description(p,
					  obs_module_text("StreamGuard.IgnoreUrls.Description"));

	p = obs_properties_add_bool(props, "use_entropy",
				    obs_module_text("StreamGuard.UseEntropy"));
	obs_property_set_long_description(p,
					  obs_module_text("StreamGuard.UseEntropy.Description"));

	p = obs_properties_add_bool(props, "use_label_proximity",
				    obs_module_text("StreamGuard.UseLabelProximity"));
	obs_property_set_long_description(
		p, obs_module_text("StreamGuard.UseLabelProximity.Description"));

	return props;
}

static void streamguard_filter_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "max_resolution", DEFAULT_MAX_DIM);
	obs_data_set_default_int(settings, "ocr_active_hz", DEFAULT_OCR_HZ);
	obs_data_set_default_int(settings, "ocr_idle_hz", DEFAULT_OCR_IDLE_HZ);
	obs_data_set_default_int(settings, "buffer_delay_ms", DEFAULT_DELAY_MS);
	obs_data_set_default_int(settings, "blur_strength", 1); /* Medium */
	obs_data_set_default_bool(settings, "ignore_urls", true);
	obs_data_set_default_bool(settings, "use_entropy", true);
	obs_data_set_default_bool(settings, "use_label_proximity", true);
}

static int clamp_int(int v, int lo, int hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

static void streamguard_filter_update(void *data, obs_data_t *settings)
{
	struct streamguard_filter *f = data;
	if (!f || !settings)
		return;

	f->pipeline_max_dim = (int)obs_data_get_int(settings, "max_resolution");

	int active_hz = clamp_int((int)obs_data_get_int(settings, "ocr_active_hz"), 1, 10);
	int idle_hz = clamp_int((int)obs_data_get_int(settings, "ocr_idle_hz"), 1, 6);
	if (idle_hz > active_hz)
		idle_hz = active_hz;
	f->ocr_active_ns = 1000000000ULL / (uint64_t)active_hz;
	f->ocr_idle_ns = 1000000000ULL / (uint64_t)idle_hz;

	int delay_ms = clamp_int((int)obs_data_get_int(settings, "buffer_delay_ms"), 300, 1500);
	int new_frames = clamp_int(delay_ms * 60 / 1000, 1, MAX_DELAY_FRAMES);
	if (new_frames != f->effective_delay_frames) {
		f->effective_delay_frames = new_frames;
		f->effective_cap = new_frames + 2;
		/* Reset the ring; in-flight slots beyond the new cap would
		 * never get reached otherwise and the warmup logic would
		 * read uninitialised slots. */
		f->write_idx = 0;
		f->read_idx = 0;
		f->items = 0;
	}

	int strength = (int)obs_data_get_int(settings, "blur_strength");
	switch (strength) {
	case 0: f->blur_radius = BLUR_LOW; break;
	case 2: f->blur_radius = BLUR_HIGH; break;
	default: f->blur_radius = BLUR_MEDIUM; break;
	}

	if (f->detector) {
		sg_detector_set_ignore_urls(f->detector,
					    obs_data_get_bool(settings, "ignore_urls"));
		sg_detector_set_use_entropy(f->detector,
					    obs_data_get_bool(settings, "use_entropy"));
		sg_detector_set_use_label_proximity(
			f->detector, obs_data_get_bool(settings, "use_label_proximity"));
	}
}

static void *streamguard_filter_create(obs_data_t *settings, obs_source_t *source)
{
	struct streamguard_filter *f = bzalloc(sizeof(struct streamguard_filter));
	f->source = source;
	f->region_ttl_ns = REGION_TTL_NS_DEFAULT;
	f->region_pad_pct = REGION_PAD_PCT_DEFAULT;
	f->effective_delay_frames = MAX_DELAY_FRAMES; /* overwritten by update() */
	f->effective_cap = BUFFER_CAP;
	f->ocr_active_ns = 1000000000ULL / DEFAULT_OCR_HZ;
	f->ocr_idle_ns = 1000000000ULL / DEFAULT_OCR_IDLE_HZ;
	f->blur_radius = BLUR_MEDIUM;
	f->pipeline_max_dim = DEFAULT_MAX_DIM;
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
		f->param_blur_radius =
			gs_effect_get_param_by_name(f->censor_effect, "blur_radius");
		obs_log(LOG_INFO,
			"censor effect loaded: image=%p mask=%p blur_radius=%p",
			(void *)f->param_image, (void *)f->param_mask,
			(void *)f->param_blur_radius);
	} else {
		obs_log(LOG_ERROR, "failed to load streamguard-censor.effect");
	}
	const uint8_t *ptrs[1] = {f->mask_buf};
	f->mask_tex = gs_texture_create(MASK_W, MASK_H, GS_R8, 1, ptrs, GS_DYNAMIC);
	obs_leave_graphics();

	f->ocr = sg_ocr_create();
	f->detector = sg_detector_create();

	streamguard_filter_update(f, settings);
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

/* Compute internal pipeline (capture + ring + OCR) dimensions, clamped to
 * the user's "Max processing resolution" setting. Aspect ratio preserved. */
static void streamguard_compute_pipeline_dims(const struct streamguard_filter *f,
					      uint32_t src_w, uint32_t src_h,
					      uint32_t *out_w, uint32_t *out_h)
{
	if (f->pipeline_max_dim <= 0) {
		*out_w = src_w;
		*out_h = src_h;
		return;
	}
	uint32_t max_dim = (uint32_t)f->pipeline_max_dim;
	if (src_w <= max_dim && src_h <= max_dim) {
		*out_w = src_w;
		*out_h = src_h;
		return;
	}
	double scale =
		(double)max_dim / (double)(src_w > src_h ? src_w : src_h);
	uint32_t scaled_w = (uint32_t)((double)src_w * scale + 0.5);
	uint32_t scaled_h = (uint32_t)((double)src_h * scale + 0.5);
	if (scaled_w < 2) scaled_w = 2;
	if (scaled_h < 2) scaled_h = 2;
	*out_w = scaled_w;
	*out_h = scaled_h;
}

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
	uint32_t src_w = target ? obs_source_get_base_width(target) : 0;
	uint32_t src_h = target ? obs_source_get_base_height(target) : 0;
	if (!target || src_w == 0 || src_h == 0 || !f->censor_effect) {
		obs_source_skip_video_filter(f->source);
		return;
	}

	uint32_t cap_w, cap_h;
	streamguard_compute_pipeline_dims(f, src_w, src_h, &cap_w, &cap_h);

	gs_texture_t *fresh = streamguard_capture_source(f, cap_w, cap_h);
	if (!fresh) {
		obs_source_skip_video_filter(f->source);
		return;
	}

	uint64_t now = os_gettime_ns();
	uint64_t interval = (f->idle_streak >= IDLE_THRESHOLD_TICKS) ? f->ocr_idle_ns
								     : f->ocr_active_ns;
	if (now - f->last_ocr_ns >= interval) {
		streamguard_submit_ocr(f, fresh, cap_w, cap_h);
		f->last_ocr_ns = now;
	}

	streamguard_prune_regions(f, now);
	streamguard_build_mask(f, now);
	gs_texture_set_image(f->mask_tex, f->mask_buf, MASK_W, false);

	static uint64_t s_last_diag_ns = 0;
	if (now - s_last_diag_ns > 2000000000ULL) {
		s_last_diag_ns = now;
		pthread_mutex_lock(&f->regions_mutex);
		int rc = f->region_count;
		pthread_mutex_unlock(&f->regions_mutex);
		obs_log(LOG_INFO,
			"render: cap=%ux%u src=%ux%u delay_frames=%d items=%d regions=%d idle=%d",
			cap_w, cap_h, src_w, src_h, f->effective_delay_frames, f->items, rc,
			f->idle_streak);
	}

	if (f->items > f->effective_delay_frames) {
		gs_texture_t *delayed = gs_texrender_get_texture(f->ring[f->read_idx]);
		if (delayed) {
			gs_effect_set_texture(f->param_image, delayed);
			gs_effect_set_texture(f->param_mask, f->mask_tex);
			if (f->param_blur_radius)
				gs_effect_set_float(f->param_blur_radius, f->blur_radius);
			while (gs_effect_loop(f->censor_effect, "Draw")) {
				/* Output at full source dims; the GPU upsamples
				 * the (possibly downscaled) ring slot. */
				gs_draw_sprite(delayed, 0, src_w, src_h);
			}
		}
		f->read_idx = (f->read_idx + 1) % f->effective_cap;
		f->items--;
	} else {
		streamguard_draw_black(src_w, src_h);
	}

	f->write_idx = (f->write_idx + 1) % f->effective_cap;
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
	.get_properties = streamguard_filter_get_properties,
	.get_defaults = streamguard_filter_get_defaults,
	.update = streamguard_filter_update,
};
