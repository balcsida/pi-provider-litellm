// Ports src/protocols.ts and normalizeBaseUrl from src/discover.ts
package protocols

import (
	"errors"
	"net"
	"net/url"
	"regexp"
	"strings"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// LiteLLMAPINames is LITELLM_API_NAMES.
var LiteLLMAPINames = []types.LiteLLMApi{types.APIAnthropicMessages, types.APIOpenAICompletions, types.APIOpenAIResponses}

var (
	trailingSlashes = regexp.MustCompile(`/+$`)
	trailingV1      = regexp.MustCompile(`(?i)/v1/?$`)
)

// IsLiteLLMAPI is isLiteLLMApi.
func IsLiteLLMAPI(api string) bool {
	for _, n := range LiteLLMAPINames {
		if string(n) == api {
			return true
		}
	}
	return false
}

// NormalizeBaseURL is normalizeBaseUrl: it requires HTTPS (HTTP only for loopback hosts or when
// allowInsecureHTTP is set), then strips trailing slashes and a trailing /v1.
func NormalizeBaseURL(input string, allowInsecureHTTP bool) (string, error) {
	// A *url.Error embeds the input, which may carry userinfo or tokens: never return it.
	u, err := url.Parse(input)
	if err != nil || u.Hostname() == "" {
		return "", errors.New("LiteLLM base URL is invalid")
	}
	host := strings.ToLower(u.Hostname())
	// Only "localhost", "::1" and dotted-quad 127.x.y.z count as loopback. The WHATWG URL parser
	// also canonicalizes "127.1" and "2130706433" to 127.0.0.1; here they fail closed.
	ip := net.ParseIP(host)
	loopback := host == "localhost" || host == "::1" || (ip != nil && ip.To4() != nil && strings.HasPrefix(host, "127."))
	if u.Scheme != "https" && !(u.Scheme == "http" && (loopback || allowInsecureHTTP)) {
		return "", errors.New("LiteLLM base URL must use HTTPS except for loopback hosts")
	}
	return trailingV1.ReplaceAllString(trailingSlashes.ReplaceAllString(input, ""), ""), nil
}

// ResolveModelBaseURL is resolveModelBaseUrl: Messages talks to the proxy root, Chat and
// Responses to <root>/v1.
func ResolveModelBaseURL(baseURL string, api types.LiteLLMApi, allowInsecureHTTP bool) (string, error) {
	root, err := NormalizeBaseURL(baseURL, allowInsecureHTTP)
	if err != nil {
		return "", err
	}
	switch api {
	case types.APIAnthropicMessages:
		return root, nil
	case types.APIOpenAICompletions, types.APIOpenAIResponses:
		return root + "/v1", nil
	}
	return "", errors.New("unknown LiteLLM api: " + string(api))
}
