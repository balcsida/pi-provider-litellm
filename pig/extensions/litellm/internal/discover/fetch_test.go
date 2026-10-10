package discover

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCapJSONDepth(t *testing.T) {
	deepArray := strings.Repeat("[", 300) + strings.Repeat("]", 300)
	deepObject := `{"a":` + strings.Repeat(`{"b":`, 300) + `1` + strings.Repeat("}", 300) + `,"c":2}`
	tests := []struct{ name, in, want string }{
		{"under limit unchanged", `{"a":[1,{"b":"x"}]}`, `{"a":[1,{"b":"x"}]}`},
		{"deep array", deepArray, strings.Repeat("[", 256) + "null" + strings.Repeat("]", 256)},
		{"deep object keeps siblings", deepObject, `{"a":` + strings.Repeat(`{"b":`, 255) + `null` + strings.Repeat("}", 255) + `,"c":2}`},
		{"braces in strings", `{"a":"{{{[[[","b":"\\"{[\\\\"}`, `{"a":"{{{[[[","b":"\\"{[\\\\"}`},
		{"unterminated untouched", strings.Repeat("[", 300), strings.Repeat("[", 300)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := string(capJSONDepth([]byte(tc.in), 256)); got != tc.want {
				t.Errorf("got %.80q want %.80q", got, tc.want)
			}
		})
	}
	// Strings full of brackets must not count toward depth.
	if got := string(capJSONDepth([]byte(`[`+strings.Repeat(`"{[",`, 500)+`1]`), 2)); strings.Contains(got, "null") {
		t.Errorf("string brackets counted: %.80q", got)
	}
}

func fetchBody(t *testing.T, body string) (FetchResult[struct {
	Data []map[string]any `json:"data"`
}], error) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return FetchJSON[struct {
		Data []map[string]any `json:"data"`
	}](context.Background(), server.URL, "", Options{})
}

func TestFetchJSON_ShapeVersusSyntax(t *testing.T) {
	for _, body := range []string{`[1,2]`, `"text"`, `{"other":1}`, `{"data":"x"}`} {
		result, err := fetchBody(t, body)
		if err != nil || !result.OK || len(result.Data.Data) != 0 {
			t.Errorf("%s: result=%+v err=%v", body, result, err)
		}
	}
	if _, err := fetchBody(t, "not json"); err == nil {
		t.Error("want error for non-JSON body")
	}
}
