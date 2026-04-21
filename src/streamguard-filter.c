/*
StreamGuard filter — MVP scaffold: passes video through untouched while we
wire up the OCR/detection/blur pipeline in follow-up steps.
*/

#include <obs-module.h>
#include <plugin-support.h>

struct streamguard_filter {
	obs_source_t *source;
};

static const char *streamguard_filter_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("StreamGuard.Filter");
}

static void *streamguard_filter_create(obs_data_t *settings, obs_source_t *source)
{
	UNUSED_PARAMETER(settings);
	struct streamguard_filter *f = bzalloc(sizeof(struct streamguard_filter));
	f->source = source;
	return f;
}

static void streamguard_filter_destroy(void *data)
{
	bfree(data);
}

static void streamguard_filter_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct streamguard_filter *f = data;
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
