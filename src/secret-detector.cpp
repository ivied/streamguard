/*
StreamGuard — secret detector implementation.

Rule set is deliberately conservative: the cost of a false positive is a
blurred chunk of screen (user can see it's blurred, retakes if needed);
the cost of a false negative is a leaked credential on stream. Tune
toward catching more, not less.
*/

#include "secret-detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

namespace {

struct Rule {
	const char *name;
	std::regex pattern;
};

std::vector<Rule> build_rules()
{
	std::vector<Rule> r;
	const auto ecma = std::regex::ECMAScript | std::regex::optimize;

	// Cloud / provider API keys ------------------------------------------
	r.push_back({"aws_access_key", std::regex(R"((?:AKIA|ASIA|AIDA|AROA|AGPA|ANPA|ANVA)[0-9A-Z]{16})", ecma)});
	r.push_back({"github_token", std::regex(R"(gh[psuoru]_[A-Za-z0-9]{36,255})", ecma)});
	r.push_back({"stripe_key", std::regex(R"((?:sk|pk|rk)_(?:test|live)_[A-Za-z0-9]{24,})", ecma)});
	r.push_back({"openai_key", std::regex(R"(sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,})", ecma)});
	r.push_back({"google_api_key", std::regex(R"(AIza[0-9A-Za-z_\-]{35})", ecma)});
	r.push_back({"slack_token", std::regex(R"(xox[abpsr]-[A-Za-z0-9\-]{10,})", ecma)});
	r.push_back({"telegram_bot_token", std::regex(R"(\d{9,10}:[A-Za-z0-9_\-]{35})", ecma)});

	// Structured credentials ---------------------------------------------
	r.push_back({"jwt", std::regex(R"(eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})", ecma)});
	r.push_back({"pem_private_key",
		     std::regex(R"(-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----)",
				ecma)});
	r.push_back({"basic_auth_url", std::regex(R"(https?://[^\s:@/]+:[^\s@/]+@)", ecma)});

	// Russia-specific ----------------------------------------------------
	r.push_back({"ru_passport", std::regex(R"(\b\d{4}[ \-]\d{6}\b)", ecma)});
	r.push_back({"ru_snils", std::regex(R"(\b\d{3}-\d{3}-\d{3}[ \-]\d{2}\b)", ecma)});

	return r;
}

double shannon_entropy(const std::string &s)
{
	if (s.empty())
		return 0.0;
	int counts[256] = {0};
	for (unsigned char c : s)
		counts[c]++;
	double n = static_cast<double>(s.size());
	double H = 0.0;
	for (int i = 0; i < 256; i++) {
		if (counts[i] == 0)
			continue;
		double p = counts[i] / n;
		H -= p * std::log2(p);
	}
	return H;
}

std::string densest_token(const std::string &s)
{
	std::string best;
	std::string cur;
	for (char c : s) {
		if (c == ' ' || c == '\t' || c == '"' || c == '\'' || c == '(' || c == ')' ||
		    c == '[' || c == ']' || c == '{' || c == '}' || c == ',' || c == ';') {
			if (cur.size() > best.size())
				best = cur;
			cur.clear();
		} else {
			cur.push_back(c);
		}
	}
	if (cur.size() > best.size())
		best = cur;
	return best;
}

// A URL-ish look (http://, https://, www.). Loose on trailing chars —
// OCR often mangles the tail, but the prefix is what we care about.
bool looks_like_url(const std::string &s)
{
	static const std::regex url_re(
		R"((?:https?://|www\.)\S{3,})",
		std::regex::ECMAScript | std::regex::optimize | std::regex::icase);
	return std::regex_search(s, url_re);
}

std::string to_lower(const std::string &s)
{
	std::string out(s);
	std::transform(out.begin(), out.end(), out.begin(),
		       [](unsigned char c) { return (char)std::tolower(c); });
	return out;
}

// Does a short-ish OCR string look like a field label for a secret value?
// We check a lower-cased copy against a small list of keywords. We cap
// length at 40 chars to avoid matching prose that happens to contain
// "password" (e.g. "reset your password via email").
bool is_secret_label(const std::string &text)
{
	if (text.empty() || text.size() > 40)
		return false;
	const std::string t = to_lower(text);

	// Latin
	static const char *latin[] = {
		"password", "passwd", "pwd", "pass:", "pass ",
		"secret",   "token",  "api key", "api_key",
		"apikey",   "access key", "access_key", "accesskey",
		"private key", "privkey", "credential", "credentials",
		"seed phrase", "recovery phrase", "recovery key",
		"pin code", "pin:",
	};
	for (const char *kw : latin) {
		if (t.find(kw) != std::string::npos)
			return true;
	}

	// Russian keywords — check against the ORIGINAL (utf-8 lowered isn't
	// reliable for Cyrillic via std::tolower, which only touches ASCII).
	static const char *ru[] = {
		"пароль", "Пароль", "ПАРОЛЬ",
		"секрет", "Секрет", "СЕКРЕТ",
		"ключ",   "Ключ",   "КЛЮЧ",
		"пин",    "Пин",    "ПИН",
		"токен",  "Токен",  "ТОКЕН",
	};
	for (const char *kw : ru) {
		if (text.find(kw) != std::string::npos)
			return true;
	}

	return false;
}

// A box worth blurring as a label's neighbour — not just whitespace, not
// itself a label, not a single short token like ":" or "—".
bool is_plausible_value(const std::string &text)
{
	if (text.size() < 3)
		return false;
	int non_space = 0;
	for (char c : text) {
		if (c != ' ' && c != '\t')
			non_space++;
	}
	return non_space >= 3 && !is_secret_label(text);
}

struct Box {
	float x, y, w, h; // Vision bottom-left-origin, normalized
};

bool vertical_overlap(const Box &a, const Box &b, float min_frac)
{
	float top_a = a.y + a.h;
	float top_b = b.y + b.h;
	float ov_lo = std::max(a.y, b.y);
	float ov_hi = std::min(top_a, top_b);
	float ov = ov_hi - ov_lo;
	if (ov <= 0.0f)
		return false;
	float min_h = std::min(a.h, b.h);
	return ov >= min_h * min_frac;
}

bool horizontal_overlap(const Box &a, const Box &b, float min_frac)
{
	float right_a = a.x + a.w;
	float right_b = b.x + b.w;
	float ov_lo = std::max(a.x, b.x);
	float ov_hi = std::min(right_a, right_b);
	float ov = ov_hi - ov_lo;
	if (ov <= 0.0f)
		return false;
	float min_w = std::min(a.w, b.w);
	return ov >= min_w * min_frac;
}

// Candidate `c` is adjacent to label `l` in one of three patterns typical
// of forms / password manager detail views:
//   - same row, to the right of the label
//   - directly below the label (label on top of field)
//   - directly above the label (field over its caption)
//
// All measured in Vision bottom-left coordinates: smaller y = visually
// lower, bigger y = visually higher.
bool is_adjacent_to_label(const Box &l, const Box &c)
{
	// Case 1: same-row / right-of
	if (vertical_overlap(l, c, 0.3f) && c.x >= l.x + l.w * 0.3f &&
	    c.x <= l.x + l.w + 0.4f) {
		return true;
	}

	const float max_vertical_gap = std::max(l.h, c.h) * 3.0f;

	// Case 2: candidate directly below label (c.y + c.h ≤ l.y)
	if (c.y + c.h <= l.y &&
	    (l.y - (c.y + c.h)) <= max_vertical_gap &&
	    horizontal_overlap(l, c, 0.2f)) {
		return true;
	}

	// Case 3: candidate directly above label (c.y ≥ l.y + l.h)
	if (c.y >= l.y + l.h &&
	    (c.y - (l.y + l.h)) <= max_vertical_gap &&
	    horizontal_overlap(l, c, 0.2f)) {
		return true;
	}

	return false;
}

} // namespace

struct sg_detector {
	std::vector<Rule> rules;
	double entropy_threshold = 4.5;
	// Dropped from 20 to 14: catches password-manager-generated strings
	// like "arj3fkx_ezn3CWE3tur" (length 19) that were previously below
	// the threshold. Accepts some extra false positives on normal code
	// identifiers as the cost.
	size_t entropy_min_len = 14;
	// When true (default), URL-looking text is not flagged by entropy or
	// label-proximity. Content-regex rules (AWS, GitHub, etc.) still
	// apply, since a leaked key could in principle appear in a URL.
	bool ignore_urls = true;
	// Master switches for the two non-regex detection paths. Regex rules
	// are always on — they're cheap and very specific.
	bool use_entropy = true;
	bool use_label_proximity = true;
};

extern "C" sg_detector *sg_detector_create(void)
{
	sg_detector *d = new sg_detector();
	d->rules = build_rules();
	return d;
}

extern "C" void sg_detector_destroy(sg_detector *d)
{
	delete d;
}

extern "C" void sg_detector_set_ignore_urls(sg_detector *d, bool value)
{
	if (d)
		d->ignore_urls = value;
}

extern "C" void sg_detector_set_use_entropy(sg_detector *d, bool value)
{
	if (d)
		d->use_entropy = value;
}

extern "C" void sg_detector_set_use_label_proximity(sg_detector *d, bool value)
{
	if (d)
		d->use_label_proximity = value;
}

extern "C" bool sg_detector_check(sg_detector *d, const char *text, const char **matched_rule)
{
	if (!d || !text)
		return false;

	const std::string s(text);
	if (s.empty())
		return false;

	for (const auto &r : d->rules) {
		if (std::regex_search(s, r.pattern)) {
			if (matched_rule)
				*matched_rule = r.name;
			return true;
		}
	}

	if (!d->use_entropy)
		return false;

	// URLs slip past the content-regex pass above but will trip entropy
	// on their random-looking path / query. Skip entropy for them when
	// the toggle is on.
	if (d->ignore_urls && looks_like_url(s))
		return false;

	std::string tok = densest_token(s);
	if (tok.size() >= d->entropy_min_len &&
	    shannon_entropy(tok) >= d->entropy_threshold) {
		if (matched_rule)
			*matched_rule = "shannon_entropy";
		return true;
	}

	return false;
}

extern "C" void sg_detector_check_all(sg_detector *d, const sg_ocr_box *boxes, int count,
				      bool *out_flags, const char **out_rules)
{
	if (!d || !boxes || !out_flags || !out_rules || count <= 0)
		return;

	// Pass 1: per-string rules.
	for (int i = 0; i < count; i++) {
		out_flags[i] = false;
		out_rules[i] = NULL;
		if (sg_detector_check(d, boxes[i].text, &out_rules[i])) {
			out_flags[i] = true;
		}
	}

	if (!d->use_label_proximity)
		return;

	// Pass 2: spatial label proximity. For every label, mark any
	// plausible-value neighbour as a secret if it isn't already flagged.
	for (int i = 0; i < count; i++) {
		if (!is_secret_label(boxes[i].text))
			continue;
		Box lb{boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h};
		for (int j = 0; j < count; j++) {
			if (j == i || out_flags[j])
				continue;
			if (!is_plausible_value(boxes[j].text))
				continue;
			if (d->ignore_urls && looks_like_url(boxes[j].text))
				continue;
			Box cb{boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h};
			if (is_adjacent_to_label(lb, cb)) {
				out_flags[j] = true;
				out_rules[j] = "label_proximity";
			}
		}
	}
}
