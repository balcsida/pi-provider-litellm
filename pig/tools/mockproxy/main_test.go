package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"testing"
)

func captureStderr(fn func()) string {
	r, w, _ := os.Pipe()
	old := os.Stderr
	os.Stderr = w
	fn()
	w.Close()
	os.Stderr = old
	var buf bytes.Buffer
	io.Copy(&buf, r)
	return buf.String()
}

func startServer(t *testing.T) (string, string) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen failed: %v", err)
	}

	testKey := "test-key-123"
	mux := http.NewServeMux()
	mux.HandleFunc("GET /model/info", makeHandleModelInfo(testKey))
	mux.HandleFunc("GET /v1/models", makeHandleModels(testKey))
	mux.HandleFunc("GET /health/liveliness", handleHealth)
	mux.HandleFunc("POST /v1/chat/completions", makeHandleChatCompletions(testKey))
	mux.HandleFunc("POST /v1/responses", makeHandleResponses(testKey))
	mux.HandleFunc("POST /v1/messages", makeHandleMessages(testKey))

	server := &http.Server{Handler: mux}
	go server.Serve(listener)

	addr := listener.Addr().String()
	return fmt.Sprintf("http://%s", addr), testKey
}

func TestUnauthorized(t *testing.T) {
	baseURL, _ := startServer(t)

	resp, err := http.Get(baseURL + "/model/info")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", resp.StatusCode)
	}

	var errResp map[string]interface{}
	_ = json.NewDecoder(resp.Body).Decode(&errResp)
	if errMsg, ok := errResp["error"].(map[string]interface{})["message"]; !ok || errMsg != "invalid key" {
		t.Errorf("expected 'invalid key' error")
	}
}

func TestModelInfo(t *testing.T) {
	baseURL, testKey := startServer(t)

	req, err := http.NewRequest("GET", baseURL+"/model/info", nil)
	if err != nil {
		t.Fatalf("request creation failed: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+testKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	var envelope struct {
		Data []ModelInfo `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	models := envelope.Data

	if len(models) != 3 {
		t.Errorf("expected 3 models, got %d", len(models))
	}

	names := map[string]bool{}
	for _, m := range models {
		names[m.ModelName] = true
	}

	for _, expected := range []string{"mock-chat", "mock-responses", "mock-claude"} {
		if !names[expected] {
			t.Errorf("model %s not found", expected)
		}
	}
}

func TestChatCompletionsStream(t *testing.T) {
	baseURL, testKey := startServer(t)

	body := map[string]interface{}{
		"model":  "mock-chat",
		"stream": true,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "test message",
			},
		},
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	req, err := http.NewRequest("POST", baseURL+"/v1/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		t.Fatalf("request creation failed: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+testKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	body_, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}
	responseText := string(body_)

	if !bytes.Contains(body_, []byte("mock repl")) || !bytes.Contains(body_, []byte("t message")) {
		t.Errorf("response missing expected text. got: %s", responseText)
	}

	if !bytes.Contains(body_, []byte("[DONE]")) {
		t.Errorf("response missing [DONE]. got: %s", responseText)
	}
}

func TestTextPartExtractionArray(t *testing.T) {
	tests := []struct {
		name    string
		content interface{}
		expect  string
	}{
		{"string", "hello world", "hello world"},
		{
			"chat array",
			[]interface{}{
				map[string]interface{}{"type": "text", "text": "hello"},
				map[string]interface{}{"type": "text", "text": " world"},
			},
			"hello world",
		},
		{
			"responses array",
			[]interface{}{
				map[string]interface{}{"type": "input_text", "text": "test"},
			},
			"test",
		},
		{"empty", nil, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractTextFromContent(tt.content)
			if result != tt.expect {
				t.Errorf("expected %q, got %q", tt.expect, result)
			}
		})
	}
}

func TestResponseHeaders(t *testing.T) {
	baseURL, testKey := startServer(t)

	tests := []struct {
		name            string
		endpoint        string
		body            map[string]interface{}
		expectEffort    string
		expectToolCount string
	}{
		{
			"chat with reasoning",
			"/v1/chat/completions",
			map[string]interface{}{
				"model":            "mock-chat",
				"reasoning_effort": "medium",
				"stream":           false,
				"messages":         []map[string]string{{"role": "user", "content": "test"}},
				"tools":            []map[string]string{{"type": "function"}, {"type": "function"}},
			},
			"medium",
			"2",
		},
		{
			"responses with reasoning",
			"/v1/responses",
			map[string]interface{}{
				"model":     "mock-responses",
				"stream":    true,
				"messages":  []map[string]string{{"role": "user", "content": "test"}},
				"reasoning": map[string]interface{}{"effort": "high"},
			},
			"high",
			"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, _ := json.Marshal(tt.body)
			req, _ := http.NewRequest("POST", baseURL+tt.endpoint, bytes.NewReader(bodyBytes))
			req.Header.Set("Authorization", "Bearer "+testKey)
			req.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			defer resp.Body.Close()

			if tt.expectEffort != "" && resp.Header.Get("x-mock-reasoning-effort") != tt.expectEffort {
				t.Errorf("expected reasoning effort %s, got %s", tt.expectEffort, resp.Header.Get("x-mock-reasoning-effort"))
			}
			if tt.expectToolCount != "" && resp.Header.Get("x-mock-tool-count") != tt.expectToolCount {
				t.Errorf("expected tool count %s, got %s", tt.expectToolCount, resp.Header.Get("x-mock-tool-count"))
			}
		})
	}
}

// deltaText is the text one streamed event carries, by protocol.
func deltaText(event map[string]interface{}) string {
	if choices, ok := event["choices"].([]interface{}); ok && len(choices) > 0 {
		delta, _ := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
		text, _ := delta["content"].(string)
		return text
	}
	switch event["type"] {
	case "content_block_delta":
		delta, _ := event["delta"].(map[string]interface{})
		text, _ := delta["text"].(string)
		return text
	case "response.output_text.delta":
		text, _ := event["delta"].(string)
		return text
	}
	return ""
}

func TestStreamsCarryQuotedAndMultilinePrompts(t *testing.T) {
	baseURL, testKey := startServer(t)
	prompt := "say \"hi\"\\\nand ünïcode — 日本語"
	messages := []map[string]string{{"role": "user", "content": prompt}}
	for _, tc := range []struct {
		path string
		body map[string]interface{}
	}{
		{"/v1/chat/completions", map[string]interface{}{"model": "mock-chat", "stream": true, "messages": messages}},
		{"/v1/messages", map[string]interface{}{"model": "mock-claude", "stream": true, "messages": messages}},
		{"/v1/responses", map[string]interface{}{"model": "mock-responses", "input": messages}},
	} {
		bodyBytes, _ := json.Marshal(tc.body)
		req, _ := http.NewRequest("POST", baseURL+tc.path, bytes.NewReader(bodyBytes))
		req.Header.Set("Authorization", "Bearer "+testKey)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s: %v", tc.path, err)
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		var got string
		for _, line := range bytes.Split(raw, []byte("\n")) {
			data, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok || string(data) == "[DONE]" {
				continue
			}
			var event map[string]interface{}
			if err := json.Unmarshal(data, &event); err != nil {
				t.Fatalf("%s: invalid stream data %s: %v", tc.path, data, err)
			}
			got += deltaText(event)
		}
		if want := "mock reply to: " + prompt; got != want {
			t.Errorf("%s: streamed %q, want %q", tc.path, got, want)
		}
	}
}
