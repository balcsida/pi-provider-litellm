// Ports the normalizeBaseUrl cases of tests/discover.test.ts and tests/probe-proxy.test.ts
package protocols

import (
	"strings"
	"testing"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func TestNormalizeBaseURL(t *testing.T) {
	errCases := []struct {
		name, in string
		insecure bool
	}{
		{"rejects insecure non-loopback endpoints", "http://litellm.example.com", false},
		{"does not allow other insecure protocols", "ftp://host.docker.internal", true},
	}
	for _, c := range errCases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := NormalizeBaseURL(c.in, c.insecure); err == nil || !strings.Contains(err.Error(), "HTTPS") {
				t.Fatalf("err = %v", err)
			}
		})
	}
	okCases := []struct {
		name, in, want string
		insecure       bool
	}{
		{"allows an explicitly configured insecure endpoint", "http://host.docker.internal/v1", "http://host.docker.internal", true},
		{"strips trailing slashes", "https://x.example.com/", "https://x.example.com", false},
		{"strips many trailing slashes", "https://x.example.com///", "https://x.example.com", false},
		{"strips a single trailing /v1 suffix", "https://x.example.com/v1", "https://x.example.com", false},
		{"strips /v1/", "https://x.example.com/v1/", "https://x.example.com", false},
		{"is case-insensitive on /v1", "https://x.example.com/V1", "https://x.example.com", false},
		{"does not strip /v2", "https://x.example.com/v2", "https://x.example.com/v2", false},
		{"does not strip /v1beta", "https://x.example.com/v1beta", "https://x.example.com/v1beta", false},
		{"preserves a base path that is not /v1", "https://x.example.com/proxy", "https://x.example.com/proxy", false},
		{"probe-proxy: strips /v1/", "https://proxy.example/v1/", "https://proxy.example", false},
		{"allows loopback http", "http://127.0.0.1:4000", "http://127.0.0.1:4000", false},
		{"allows localhost http", "http://localhost:4000/", "http://localhost:4000", false},
		{"allows IPv6 loopback http", "http://[::1]:4000", "http://[::1]:4000", false},
	}
	for _, c := range okCases {
		t.Run(c.name, func(t *testing.T) {
			got, err := NormalizeBaseURL(c.in, c.insecure)
			if err != nil || got != c.want {
				t.Fatalf("got %q, %v want %q", got, err, c.want)
			}
		})
	}
}

func TestResolveModelBaseURL(t *testing.T) {
	cases := map[types.LiteLLMApi]string{
		types.APIAnthropicMessages: "https://x.example.com",
		types.APIOpenAICompletions: "https://x.example.com/v1",
		types.APIOpenAIResponses:   "https://x.example.com/v1",
	}
	for api, want := range cases {
		got, err := ResolveModelBaseURL("https://x.example.com/v1/", api, false)
		if err != nil || got != want {
			t.Errorf("%s: got %q, %v want %q", api, got, err, want)
		}
	}
	if _, err := ResolveModelBaseURL("http://litellm.example.com", types.APIOpenAIResponses, false); err == nil {
		t.Error("expected HTTPS error")
	}
}

func TestIsLiteLLMAPI(t *testing.T) {
	for _, n := range LiteLLMAPINames {
		if !IsLiteLLMAPI(string(n)) {
			t.Errorf("%s", n)
		}
	}
	for _, n := range []string{"", "openai", "toString", "anthropic"} {
		if IsLiteLLMAPI(n) {
			t.Errorf("%q", n)
		}
	}
}
