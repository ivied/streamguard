/*
StreamGuard — secret detector.

Given a piece of text, returns whether it looks like a sensitive token
(API key, password, private key material, РФ passport/СНИЛС, etc.) and
which rule matched. Used to decide whether to blur an OCR bounding box.
*/

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sg_detector sg_detector;

sg_detector *sg_detector_create(void);
void sg_detector_destroy(sg_detector *d);

/*
 * Returns true if `text` matches any enabled rule. When it does and
 * `matched_rule` is non-NULL, *matched_rule is set to a static C string
 * naming the rule (e.g. "aws_access_key", "shannon_entropy"). The pointer
 * is owned by the detector; do not free it.
 */
bool sg_detector_check(sg_detector *d, const char *text, const char **matched_rule);

#ifdef __cplusplus
}
#endif
