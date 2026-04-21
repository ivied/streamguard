/*
StreamGuard — secret detector implementation.

Rule set is deliberately conservative: the cost of a false positive is a
blurred chunk of screen (user can see it's blurred, retakes if needed);
the cost of a false negative is a leaked credential on stream. Tune
toward catching more, not less.
*/

#include "secret-detector.h"

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
	// std::regex::ECMAScript + case-sensitive unless noted.
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
	// Passport (4 digits serial + 6 digits number, optional space/dash)
	r.push_back({"ru_passport", std::regex(R"(\b\d{4}[ \-]?\d{6}\b)", ecma)});
	// СНИЛС: 3-3-3 12 (space or dash before last two)
	r.push_back({"ru_snils", std::regex(R"(\b\d{3}-\d{3}-\d{3}[ \-]\d{2}\b)", ecma)});

	return r;
}

// Shannon entropy in bits per character over the printable payload.
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

// Return the "densest" token: longest contiguous run of non-space characters.
// We run entropy only on that, not the whole phrase (`const x = "..."` has
// low entropy overall even if the literal inside is high-entropy).
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

} // namespace

struct sg_detector {
	std::vector<Rule> rules;
	// Shannon threshold in bits/char. 4.5 is roughly the cutoff where
	// base64/hex/random identifiers live; English words sit around 3.5–4.0.
	double entropy_threshold = 4.5;
	// Minimum token length to even consider for entropy — below this,
	// short random-looking English words cause too many false positives.
	size_t entropy_min_len = 20;
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

	std::string tok = densest_token(s);
	if (tok.size() >= d->entropy_min_len &&
	    shannon_entropy(tok) >= d->entropy_threshold) {
		if (matched_rule)
			*matched_rule = "shannon_entropy";
		return true;
	}

	return false;
}
