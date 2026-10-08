package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"testing"
)

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
