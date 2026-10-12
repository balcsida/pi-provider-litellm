package discover

// Ports the probeProxyVersion cases of tests/proxy-version.test.ts. The parse and ordering cases live
// with internal/proxyversion.

import (
	"context"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/proxyversion"
)

// spyBody records whether anything read the reply body.
type spyBody struct{ read bool }

func (b *spyBody) Read([]byte) (int, error) {
	b.read = true
	return 0, io.EOF
}

func (b *spyBody) Close() error { return nil }

func probeReply(status int, body io.ReadCloser, headers map[string]string) *http.Response {
	header := http.Header{}
	for name, value := range headers {
		header.Set(name, value)
	}
	return &http.Response{StatusCode: status, Header: header, Body: body}
}

func TestProbeProxyVersion(t *testing.T) {
	t.Run("reads the version from the error reply to a made-up Responses id", func(t *testing.T) {
		var seen []*http.Request
		client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			seen = append(seen, r)
			return probeReply(400, io.NopCloser(strings.NewReader(`{"error":{"message":"no healthy deployments"}}`)),
				map[string]string{"x-litellm-version": "1.103.0"}), nil
		})}
		options := Options{HTTPClient: client}
		options.Headers = map[string]string{"x-tenant": "a"}

		got := ProbeProxyVersion(context.Background(), "https://proxy.test", "sk-test", options)

		if want := (&proxyversion.Version{Major: 1, Minor: 103, Patch: 0}); !reflect.DeepEqual(got, want) {
			t.Fatalf("version = %+v, want %+v", got, want)
		}
		if len(seen) != 1 {
			t.Fatalf("requests = %d, want 1", len(seen))
		}
		request := seen[0]
		if request.URL.String() != "https://proxy.test/v1/responses/resp_version_probe" || request.Method != http.MethodGet {
			t.Fatalf("request = %s %s", request.Method, request.URL)
		}
		if request.Header.Get("x-tenant") != "a" || request.Header.Get("Authorization") != "Bearer sk-test" {
			t.Fatalf("headers = %v", request.Header)
		}
	})

	for _, tc := range []struct {
		name    string
		respond func(*http.Request) (*http.Response, error)
	}{
		{"a reply without the header", func(*http.Request) (*http.Response, error) {
			return probeReply(400, io.NopCloser(strings.NewReader("{}")), nil), nil
		}},
		{"an unreadable header", func(*http.Request) (*http.Response, error) {
			return probeReply(400, io.NopCloser(strings.NewReader("{}")), map[string]string{"x-litellm-version": "latest"}), nil
		}},
		{"a refused connection", func(*http.Request) (*http.Response, error) { return nil, errors.New("fetch failed") }},
		{"an aborted request", func(r *http.Request) (*http.Response, error) { return nil, context.Canceled }},
	} {
		t.Run("withholds a version on "+tc.name, func(t *testing.T) {
			options := Options{HTTPClient: &http.Client{Transport: roundTripFunc(tc.respond)}}
			if got := ProbeProxyVersion(context.Background(), "https://proxy.test", "sk-test", options); got != nil {
				t.Fatalf("version = %+v, want nil", got)
			}
		})
	}

	t.Run("does not read the reply body", func(t *testing.T) {
		body := &spyBody{}
		options := Options{HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return probeReply(400, body, map[string]string{"x-litellm-version": "1.103.0"}), nil
		})}}

		ProbeProxyVersion(context.Background(), "https://proxy.test", "sk-test", options)

		if body.read {
			t.Fatal("the probe read the reply body")
		}
	})
}
