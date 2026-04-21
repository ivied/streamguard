/*
StreamGuard — Apple Vision OCR C API wrapper.

Single serial queue internally; sg_ocr_submit drops the frame if a previous
request is still in flight (we don't need realtime OCR — losing one frame at
2 Hz is fine).

Bounding boxes are normalized to [0, 1] with Vision-style origin at
*bottom-left* of the image. Convert to top-left coordinates with:
    y_top = 1.0 - y - h
*/

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sg_ocr_box {
	float x;
	float y;
	float w;
	float h;
	float confidence;
	char text[512];
} sg_ocr_box;

typedef struct sg_ocr_result {
	sg_ocr_box *boxes;
	int count;
	uint64_t frame_id;
} sg_ocr_result;

typedef void (*sg_ocr_callback)(sg_ocr_result *result, void *user_data);

typedef struct sg_ocr_ctx sg_ocr_ctx;

sg_ocr_ctx *sg_ocr_create(void);
void sg_ocr_destroy(sg_ocr_ctx *ctx);

bool sg_ocr_submit(sg_ocr_ctx *ctx, const uint8_t *bgra, int width, int height, int bytes_per_row,
		   uint64_t frame_id, sg_ocr_callback callback, void *user_data);

void sg_ocr_free_result(sg_ocr_result *result);

#ifdef __cplusplus
}
#endif
