// Ports tests/proxy-version.test.ts (parseProxyVersion, proxyVersionAtLeast; probe I/O lives in discover)
package proxyversion

import (
	"net/http"
	"strings"
	"testing"
)

func TestParseProxyVersion(t *testing.T) {
	reads := map[string]Version{
		"1.102.0":       {1, 102, 0, false},
		"v1.103.0":      {1, 103, 0, false},
		" 1.103.1 ":     {1, 103, 1, false},
		"1.103.0-rc.1":  {1, 103, 0, true},
		"1.103.0rc1":    {1, 103, 0, true},
		"1.103.0.dev2":  {1, 103, 0, true},
		"1.103.0.post1": {1, 103, 0, true},
	}
	for in, want := range reads {
		if got := Parse(in); got == nil || *got != want {
			t.Errorf("reads %q: got %+v want %+v", in, got, want)
		}
	}
	for _, in := range []string{"", "latest", "1.103", "1", "one.two.three", "1.103.0-" + strings.Repeat("x", 80), "999999.0.0"} {
		if got := Parse(in); got != nil {
			t.Errorf("withholds %q: got %+v", in, got)
		}
	}
}

func TestProxyVersionAtLeast(t *testing.T) {
	cases := map[string]bool{
		"1.103.0": true, "1.103.1": true, "1.104.0": true, "2.0.0": true, "1.104.0-rc.1": true,
		"1.102.1": false, "1.102.99": false, "0.999.999": false, "1.103.0-rc.1": false, "1.103.0.dev2": false,
	}
	for in, want := range cases {
		if got := AtLeast(Parse(in), Floor{1, 103, 0}); got != want {
			t.Errorf("orders %s against 1.103.0: got %v", in, got)
		}
	}
	if AtLeast(nil, Floor{0, 0, 0}) {
		t.Error("an unknown version must be older than any minimum")
	}
}

func TestFromHeader(t *testing.T) {
	h := http.Header{}
	if FromHeader(h) != nil {
		t.Error("missing header")
	}
	h.Set("X-LiteLLM-Version", "latest")
	if FromHeader(h) != nil {
		t.Error("unreadable header")
	}
	h.Set("X-LiteLLM-Version", "1.103.0")
	if got := FromHeader(h); got == nil || *got != (Version{1, 103, 0, false}) {
		t.Errorf("got %+v", got)
	}
}
