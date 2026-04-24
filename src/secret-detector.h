/*
StreamGuard — secret detector.

Two entry points:

- sg_detector_check() — per-string check: regex patterns and Shannon
  entropy. Stateless across calls.

- sg_detector_check_all() — batch check over a full OCR frame. On top of
  the per-string rules it runs a spatial pass: any short label box
  containing a keyword like "password" / "пароль" / "token" marks the
  neighbouring value box (same row, to the right) as a secret. This is
  how we catch weak / unremarkable-looking passwords ("qwerty12345"):
  by their context, not by their content.
*/

#pragma once

#include <stdbool.h>

#include "vision-ocr.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sg_detector sg_detector;

sg_detector *sg_detector_create(void);
void sg_detector_destroy(sg_detector *d);

/*
 * Toggles. Safe to call from any thread at any time — each setter flips a
 * single bool in the detector struct; readers pick it up on the next check.
 */
void sg_detector_set_ignore_urls(sg_detector *d, bool value);

bool sg_detector_check(sg_detector *d, const char *text, const char **matched_rule);

/*
 * Check all OCR boxes in one pass, filling `out_flags[i]` and `out_rules[i]`
 * (each must point to an array of `count` elements). `out_rules[i]` gets
 * a static string like "aws_access_key" / "label_proximity" owned by the
 * detector — do not free.
 */
void sg_detector_check_all(sg_detector *d, const sg_ocr_box *boxes, int count,
			   bool *out_flags, const char **out_rules);

#ifdef __cplusplus
}
#endif
