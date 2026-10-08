// Ports src/proxy-version.ts (parsing and comparison only; the HTTP probe belongs to discover).
package proxyversion

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	DefaultTimeout = 5 * time.Second
	// Header is the response header LiteLLM stamps its version on.
	Header = "x-litellm-version"
	// ProbePath is a made-up Responses id: the error reply still carries Header and calls no model.
	ProbePath = "/v1/responses/resp_version_probe"

	maxVersionLength = 64
)

var versionPattern = regexp.MustCompile(`(?s)^v?(\d{1,5})\.(\d{1,5})\.(\d{1,5})(.*)$`)

// Version keeps only the parsed numbers of a proxy-supplied version.
type Version struct {
	Major int `json:"major"`
	Minor int `json:"minor"`
	Patch int `json:"patch"`
	// Prerelease is any suffix (rc, dev, post, local build). It orders before the release it
	// names, so a pre-release never claims a fix that only the release is known to carry.
	Prerelease bool `json:"prerelease"`
}

// Floor is ProxyVersionFloor: major, minor, patch.
type Floor [3]int

// Parse is parseProxyVersion; nil means unknown.
func Parse(value string) *Version {
	if utf8.RuneCountInString(value) > maxVersionLength {
		return nil
	}
	m := versionPattern.FindStringSubmatch(strings.TrimSpace(value))
	if m == nil {
		return nil
	}
	major, _ := strconv.Atoi(m[1])
	minor, _ := strconv.Atoi(m[2])
	patch, _ := strconv.Atoi(m[3])
	return &Version{Major: major, Minor: minor, Patch: patch, Prerelease: m[4] != ""}
}

// FromHeader reads the version from a probe reply's headers.
func FromHeader(h http.Header) *Version {
	return Parse(h.Get(Header))
}

// AtLeast is proxyVersionAtLeast: an unknown version satisfies no floor.
func AtLeast(v *Version, floor Floor) bool {
	if v == nil {
		return false
	}
	actual := [3]int{v.Major, v.Minor, v.Patch}
	for i, required := range floor {
		if actual[i] != required {
			return actual[i] > required
		}
	}
	return !v.Prerelease
}
