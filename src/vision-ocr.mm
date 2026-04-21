/*
StreamGuard — Apple Vision OCR implementation.
*/

#include "vision-ocr.h"

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <CoreVideo/CoreVideo.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

struct sg_ocr_ctx {
	dispatch_queue_t queue;
	std::atomic<bool> busy;
};

extern "C" sg_ocr_ctx *sg_ocr_create(void)
{
	sg_ocr_ctx *ctx = new sg_ocr_ctx();
	ctx->queue = dispatch_queue_create("com.sergeykurdyuk.streamguard.ocr",
					   dispatch_queue_attr_make_with_qos_class(
						   DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
	ctx->busy.store(false);
	return ctx;
}

extern "C" void sg_ocr_destroy(sg_ocr_ctx *ctx)
{
	if (!ctx)
		return;
	// Drain any pending work before tearing down.
	dispatch_sync(ctx->queue, ^{
	});
	delete ctx;
}

extern "C" void sg_ocr_free_result(sg_ocr_result *result)
{
	if (!result)
		return;
	free(result->boxes);
	free(result);
}

extern "C" bool sg_ocr_submit(sg_ocr_ctx *ctx, const uint8_t *bgra, int width, int height,
			      int bytes_per_row, uint64_t frame_id, sg_ocr_callback callback,
			      void *user_data)
{
	if (!ctx || !bgra || width <= 0 || height <= 0 || !callback)
		return false;

	bool expected = false;
	if (!ctx->busy.compare_exchange_strong(expected, true)) {
		return false; // previous request still running — drop this frame
	}

	// Copy pixels so the source buffer can be released/reused immediately.
	size_t buf_size = (size_t)bytes_per_row * (size_t)height;
	uint8_t *pixels = (uint8_t *)malloc(buf_size);
	if (!pixels) {
		ctx->busy.store(false);
		return false;
	}
	memcpy(pixels, bgra, buf_size);

	dispatch_async(ctx->queue, ^{
		CVPixelBufferRef pb = NULL;
		CVReturn rc = CVPixelBufferCreateWithBytes(
			kCFAllocatorDefault, (size_t)width, (size_t)height,
			kCVPixelFormatType_32BGRA, pixels, (size_t)bytes_per_row,
			[](void *releaseRefCon, const void *baseAddress) {
				(void)releaseRefCon;
				free((void *)baseAddress);
			},
			NULL, NULL, &pb);

		if (rc != kCVReturnSuccess || !pb) {
			free(pixels);
			sg_ocr_result *empty = (sg_ocr_result *)calloc(1, sizeof(sg_ocr_result));
			empty->frame_id = frame_id;
			callback(empty, user_data);
			ctx->busy.store(false);
			return;
		}

		VNRecognizeTextRequest *req = [[VNRecognizeTextRequest alloc]
			initWithCompletionHandler:^(VNRequest *request, NSError *error) {
				sg_ocr_result *result =
					(sg_ocr_result *)calloc(1, sizeof(sg_ocr_result));
				result->frame_id = frame_id;

				if (error || !request.results) {
					callback(result, user_data);
					ctx->busy.store(false);
					return;
				}

				NSArray<VNRecognizedTextObservation *> *obs = request.results;
				int count = (int)obs.count;
				if (count > 0) {
					result->boxes = (sg_ocr_box *)calloc(
						(size_t)count, sizeof(sg_ocr_box));
					int written = 0;
					for (VNRecognizedTextObservation *o in obs) {
						VNRecognizedText *top =
							[[o topCandidates:1] firstObject];
						if (!top)
							continue;
						sg_ocr_box *b = &result->boxes[written++];
						b->x = (float)o.boundingBox.origin.x;
						b->y = (float)o.boundingBox.origin.y;
						b->w = (float)o.boundingBox.size.width;
						b->h = (float)o.boundingBox.size.height;
						b->confidence = top.confidence;
						NSString *s = top.string;
						const char *c = s.UTF8String;
						if (c) {
							strncpy(b->text, c, sizeof(b->text) - 1);
							b->text[sizeof(b->text) - 1] = '\0';
						}
					}
					result->count = written;
				}

				callback(result, user_data);
				ctx->busy.store(false);
			}];

		req.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
		req.usesLanguageCorrection = NO;
		// Leave .revision at default — Vision picks the newest revision
		// supported by the runtime, which is what we want across macOS
		// versions without gating via @available.

		VNImageRequestHandler *handler =
			[[VNImageRequestHandler alloc] initWithCVPixelBuffer:pb
								     options:@{}];
		NSError *err = nil;
		[handler performRequests:@[ req ] error:&err];

		CVPixelBufferRelease(pb);
		(void)err;
	});

	return true;
}
