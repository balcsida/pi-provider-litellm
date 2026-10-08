package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

var (
	addr   = flag.String("addr", "127.0.0.1:0", "listen address")
	key    = flag.String("key", "", "required bearer token")
	models = flag.String("models", "", "path to JSON models file (default built-in)")
	dump   = flag.Bool("dump", false, "detailed request/response logging")
)

type ModelInfo struct {
	ModelName         string      `json:"model_name"`
	LiteLLMParams     interface{} `json:"litellm_params"`
	ModelInfo         interface{} `json:"model_info"`
	CustomLLMProvider string      `json:"custom_llm_provider,omitempty"`
}

type ListResponse struct {
	Data []map[string]interface{} `json:"data"`
}

var defaultModels = []ModelInfo{
	{
		ModelName: "mock-chat",
		LiteLLMParams: map[string]interface{}{
			"model": "openai/gpt-4.1-mini",
		},
		ModelInfo: map[string]interface{}{
			"mode":                  "chat",
			"supported_endpoints":   []string{"/v1/chat/completions"},
			"max_input_tokens":      128000,
			"max_output_tokens":     16384,
			"input_cost_per_token":  1e-6,
			"output_cost_per_token": 2e-6,
			"supports_reasoning":    false,
		},
	},
	{
		ModelName: "mock-responses",
		LiteLLMParams: map[string]interface{}{
			"model": "openai/gpt-5-mini",
		},
		ModelInfo: map[string]interface{}{
			"mode":                    "responses",
			"supported_endpoints":     []string{"/v1/responses"},
			"supports_reasoning":      true,
			"reasoning_effort_levels": []string{"low", "medium", "high"},
		},
	},
	{
		ModelName: "mock-claude",
		LiteLLMParams: map[string]interface{}{
			"model": "anthropic/claude-sonnet-4-5",
		},
		CustomLLMProvider: "anthropic",
		ModelInfo: map[string]interface{}{
			"mode":                "chat",
			"supported_endpoints": []string{"/v1/messages", "/v1/chat/completions"},
			"max_input_tokens":    200000,
			"max_output_tokens":   64000,
		},
	},
}

func auth(w http.ResponseWriter, r *http.Request, checkKey string) bool {
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") || strings.TrimPrefix(auth, "Bearer ") != checkKey {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]interface{}{"error": map[string]string{"message": "invalid key"}})
		return false
	}
	return true
}

func header(w http.ResponseWriter) {
	w.Header().Set("x-litellm-version", "1.80.0")
}

func getModels() []ModelInfo {
	if *models != "" {
		data, err := os.ReadFile(*models)
		if err != nil {
			return defaultModels
		}
		var m []ModelInfo
		if err := json.Unmarshal(data, &m); err != nil {
			return defaultModels
		}
		return m
	}
	return defaultModels
}

func makeHandleModelInfo(checkKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r, checkKey) {
			return
		}
		header(w)
		w.Header().Set("Content-Type", "application/json")
		// LiteLLM wraps deployments in a data envelope: {"data": [...]}.
		json.NewEncoder(w).Encode(map[string]interface{}{"data": getModels()})
	}
}

func makeHandleModels(checkKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r, checkKey) {
			return
		}
		header(w)
		w.Header().Set("Content-Type", "application/json")
		m := getModels()
		data := make([]map[string]interface{}, len(m))
		for i, mod := range m {
			data[i] = map[string]interface{}{
				"id":       mod.ModelName,
				"object":   "model",
				"owned_by": "openai",
			}
		}
		json.NewEncoder(w).Encode(ListResponse{Data: data})
	}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	header(w)
	w.Write([]byte("I'm alive!"))
}

func extractTextFromContent(content interface{}) string {
	if str, ok := content.(string); ok {
		return str
	}
	if arr, ok := content.([]interface{}); ok {
		var parts []string
		for _, part := range arr {
			if obj, ok := part.(map[string]interface{}); ok {
				if t, ok := obj["type"].(string); ok && (t == "text" || t == "input_text") {
					if text, ok := obj["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "")
	}
	return ""
}

func getLastUserMessage(msgs interface{}) string {
	if arr, ok := msgs.([]interface{}); ok {
		for i := len(arr) - 1; i >= 0; i-- {
			if msg, ok := arr[i].(map[string]interface{}); ok {
				if role, ok := msg["role"].(string); ok && role == "user" {
					if content, ok := msg["content"].(interface{}); ok {
						return extractTextFromContent(content)
					}
				}
			}
		}
	}
	return ""
}

func makeHandleChatCompletions(checkKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r, checkKey) {
			return
		}
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)

		header(w)
		w.Header().Set("x-litellm-response-cost", "0.000123")
		if v, ok := req["reasoning_effort"].(string); ok && v != "" {
			w.Header().Set("x-mock-reasoning-effort", v)
		}
		if tools, ok := req["tools"].([]interface{}); ok {
			w.Header().Set("x-mock-tool-count", fmt.Sprintf("%d", len(tools)))
		}

		stream, _ := req["stream"].(bool)
		lastMsg := getLastUserMessage(req["messages"])
		replyText := "mock reply to: " + lastMsg

		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			chunks := []string{
				`{"choices":[{"delta":{"role":"assistant"},"index":0}]}`,
				fmt.Sprintf(`{"choices":[{"delta":{"content":"%s"},"index":0}]}`, replyText[:len(replyText)/3]),
				fmt.Sprintf(`{"choices":[{"delta":{"content":"%s"},"index":0}]}`, replyText[len(replyText)/3:2*len(replyText)/3]),
				fmt.Sprintf(`{"choices":[{"delta":{"content":"%s"},"index":0}]}`, replyText[2*len(replyText)/3:]),
				`{"choices":[{"delta":{"content":""},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`,
			}
			for _, chunk := range chunks {
				fmt.Fprintf(w, "data: %s\n\n", chunk)
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
		} else {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"choices": []map[string]interface{}{
					{
						"message":       map[string]string{"role": "assistant", "content": replyText},
						"finish_reason": "stop",
						"index":         0,
					},
				},
				"usage": map[string]int{
					"prompt_tokens":     10,
					"completion_tokens": 20,
					"total_tokens":      30,
				},
			})
		}
	}
}

func makeHandleResponses(checkKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r, checkKey) {
			return
		}
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)

		header(w)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		if reasoning, ok := req["reasoning"].(map[string]interface{}); ok {
			if v, ok := reasoning["effort"].(string); ok && v != "" {
				w.Header().Set("x-mock-reasoning-effort", v)
			}
		}
		if tools, ok := req["tools"].([]interface{}); ok {
			w.Header().Set("x-mock-tool-count", fmt.Sprintf("%d", len(tools)))
		}

		lastMsg := getLastUserMessage(req["messages"])
		replyText := "mock reply to: " + lastMsg

		events := []string{
			`{"type":"response.created","response":{"id":"resp_test","object":"realtime.response","status":"in_progress","status_details":null,"created_at":1234567890,"output":[]},"event_id":"event_1"}`,
			`{"type":"response.output_item.added","response":{"output":[{"type":"message","id":"msg_test"}]},"event_id":"event_2"}`,
			`{"type":"response.content_part.added","response":{"output":[{"id":"msg_test","content":[{"type":"text","text":""}]}]},"event_id":"event_3"}`,
			fmt.Sprintf(`{"type":"response.output_text.delta","response":{"output":[{"id":"msg_test","content":[{"text":"%s"}]}]},"event_id":"event_4"}`, replyText[:len(replyText)/3]),
			fmt.Sprintf(`{"type":"response.output_text.delta","response":{"output":[{"id":"msg_test","content":[{"text":"%s"}]}]},"event_id":"event_5"}`, replyText[len(replyText)/3:2*len(replyText)/3]),
			fmt.Sprintf(`{"type":"response.output_text.delta","response":{"output":[{"id":"msg_test","content":[{"text":"%s"}]}]},"event_id":"event_6"}`, replyText[2*len(replyText)/3:]),
			`{"type":"response.output_text.done","response":{},"event_id":"event_7"}`,
			`{"type":"response.content_part.done","response":{},"event_id":"event_8"}`,
			`{"type":"response.output_item.done","response":{},"event_id":"event_9"}`,
			`{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":20}},"event_id":"event_10"}`,
		}

		for _, event := range events {
			fmt.Fprintf(w, "data: %s\n\n", event)
		}
	}
}

func makeHandleMessages(checkKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r, checkKey) {
			return
		}
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)

		header(w)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		if v, ok := req["reasoning_effort"].(string); ok && v != "" {
			w.Header().Set("x-mock-reasoning-effort", v)
		}
		if tools, ok := req["tools"].([]interface{}); ok {
			w.Header().Set("x-mock-tool-count", fmt.Sprintf("%d", len(tools)))
		}

		lastMsg := getLastUserMessage(req["messages"])
		replyText := "mock reply to: " + lastMsg

		events := []string{
			`{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"%s"}}`, replyText[:len(replyText)/3]),
			fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"%s"}}`, replyText[len(replyText)/3:2*len(replyText)/3]),
			fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"%s"}}`, replyText[2*len(replyText)/3:]),
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}`,
			`{"type":"message_stop"}`,
		}

		for _, event := range events {
			fmt.Fprintf(w, "data: %s\n\n", event)
		}
	}
}

func logRequest(r *http.Request, body []byte) {
	if !*dump {
		return
	}
	auth := "absent"
	if r.Header.Get("Authorization") != "" {
		auth = "present"
	}
	fmt.Fprintf(os.Stderr, "[DUMP] %s %s\n", r.Method, r.RequestURI)
	fmt.Fprintf(os.Stderr, "  Authorization: %s\n", auth)
	fmt.Fprintf(os.Stderr, "  x-litellm-session-id: %s\n", r.Header.Get("x-litellm-session-id"))
	fmt.Fprintf(os.Stderr, "  content-type: %s\n", r.Header.Get("content-type"))
	if r.Method == "POST" && len(body) > 0 {
		var pretty interface{}
		if err := json.Unmarshal(body, &pretty); err == nil {
			if b, err := json.MarshalIndent(pretty, "  ", "  "); err == nil {
				fmt.Fprintf(os.Stderr, "  body:\n%s\n", string(b))
			}
		}
	}
}

type dumpMiddleware struct {
	handler http.Handler
}

func (m *dumpMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if *dump && r.Method == "POST" {
		body, _ := io.ReadAll(r.Body)
		logRequest(r, body)
		r.Body = io.NopCloser(bytes.NewReader(body))
	} else if *dump {
		logRequest(r, nil)
	}
	m.handler.ServeHTTP(w, r)
}

func main() {
	flag.Parse()

	if *key == "" {
		os.Exit(1)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /model/info", makeHandleModelInfo(*key))
	mux.HandleFunc("GET /v1/models", makeHandleModels(*key))
	mux.HandleFunc("GET /health/liveliness", handleHealth)
	mux.HandleFunc("POST /v1/chat/completions", makeHandleChatCompletions(*key))
	mux.HandleFunc("POST /v1/responses", makeHandleResponses(*key))
	mux.HandleFunc("POST /v1/messages", makeHandleMessages(*key))

	var handler http.Handler = mux
	if *dump {
		handler = &dumpMiddleware{handler: mux}
	}

	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		os.Exit(1)
	}
	actualAddr := listener.Addr().String()
	fmt.Printf("listening on http://%s\n", actualAddr)

	server := &http.Server{
		Handler:     handler,
		IdleTimeout: 5 * time.Second,
	}

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		server.Close()
	}()

	if !*dump {
		wrappedListener := &logListener{Listener: listener}
		server.Serve(wrappedListener)
	} else {
		server.Serve(listener)
	}
}

type logListener struct {
	net.Listener
}

func (l *logListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	return &logConn{Conn: conn}, nil
}

type logConn struct {
	net.Conn
}

func (c *logConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	if n > 0 && len(p) > 0 {
		if bytes := p[:n]; bytes[0] >= 'A' && bytes[0] <= 'Z' {
			parts := strings.Fields(string(bytes[:100]))
			if len(parts) >= 2 {
				fmt.Fprintf(os.Stderr, "%s %s\n", parts[0], parts[1])
			}
		}
	}
	return n, err
}
