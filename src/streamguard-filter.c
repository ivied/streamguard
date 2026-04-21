/*
StreamGuard filter — video filter that periodically reads back frames from
the GPU, runs Apple Vision OCR on them, and (for now) logs recognized text.
Blur/detection pipeline is layered on top in later steps.
*/

#include <obs-module.h>
#include <plugin-support.h>
#include <util/platform.h>
#include <util/threading.h>

#include "vision-ocr.h"
#include "secret-detector.h"

#define OCR_INTERVAL_NS_DEFAULT 500000000ULL /* 500 ms = 2 Hz */

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
};

static const char *streamguard_filter_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("StreamGuard.Filter");
}

static void streamguard_ocr_done(sg_ocr_result *result, void *user_data)
{
	struct streamguard_filter *f = user_data;
	if (!result)
		return;

	int hits = 0;
	for (int i = 0; i < result->count; i++) {
		sg_ocr_box *b = &result->boxes[i];
		const char *rule = NULL;
		if (f && f->detector && sg_detector_check(f->detector, b->text, &rule)) {
			hits++;
			obs_log(LOG_WARNING,
				"[streamguard] SECRET frame=%llu rule=%s bbox=(%.2f,%.2f %.2fx%.2f) text=\"%s\"",
				(unsigned long long)result->frame_id, rule ? rule : "?", b->x,
				b->y, b->w, b->h, b->text);
		}
	}
	if (hits > 0) {
		obs_log(LOG_WARNING, "[streamguard] frame %llu: %d/%d boxes flagged",
			(unsigned long long)result->frame_id, hits, result->count);
	}
	sg_ocr_free_result(result);
}

static void *streamguard_filter_create(obs_data_t *settings, obs_source_t *source)
{
	UNUSED_PARAMETER(settings);
	struct streamguard_filter *f = bzalloc(sizeof(struct streamguard_filter));
	f->source = source;
	f->ocr_interval_ns = OCR_INTERVAL_NS_DEFAULT;
	f->last_ocr_ns = 0;
	f->next_frame_id = 0;

	obs_enter_graphics();
	f->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
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
	obs_leave_graphics();

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

	obs_source_skip_video_filter(f->source);
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
