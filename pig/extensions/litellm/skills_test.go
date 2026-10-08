package litellm

// Ports tests/skills.test.ts and tests/skills-hook-errors.test.ts.

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

type recordedRequest struct {
	Method, Path, Auth, Tenant, Accept, Body string
}

// skillsServer answers each request with the next scripted status/body (the last repeats) and records it.
type scriptedReply struct {
	status int
	body   string
}

func skillsServer(t *testing.T, replies ...scriptedReply) (*httptest.Server, func() []recordedRequest) {
	t.Helper()
	var mu sync.Mutex
	var seen []recordedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		seen = append(seen, recordedRequest{r.Method, r.URL.Path, r.Header.Get("Authorization"), r.Header.Get("x-tenant"), r.Header.Get("Accept"), string(body)})
		reply := replies[min(len(seen)-1, len(replies)-1)]
		mu.Unlock()
		if reply.status == -1 { // drop the connection
			conn, _, _ := w.(http.Hijacker).Hijack()
			conn.Close()
			return
		}
		w.WriteHeader(reply.status)
		_, _ = w.Write([]byte(reply.body))
	}))
	t.Cleanup(func() { server.Close(); resetSkillsCache() })
	resetSkillsCache()
	return server, func() []recordedRequest { mu.Lock(); defer mu.Unlock(); return append([]recordedRequest(nil), seen...) }
}

func paths(requests []recordedRequest) []string {
	var out []string
	for _, r := range requests {
		out = append(out, r.Method+" "+r.Path)
	}
	return out
}

func names(skills []types.LiteLLMSkill) []string {
	out := []string{}
	for _, s := range skills {
		out = append(out, s.Name)
	}
	return out
}

func mustList(t *testing.T, base string) []types.LiteLLMSkill {
	t.Helper()
	skills, err := listSkills(context.Background(), base, "sk-test", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	return skills
}

func TestListSkills(t *testing.T) {
	t.Run("requires the insecure flag for a non-loopback http endpoint", func(t *testing.T) {
		if _, err := listSkills(context.Background(), "http://host.docker.internal", "sk-test", nil, false); err == nil {
			t.Fatal("expected insecure endpoint rejection")
		}
		if root, err := protocols.NormalizeBaseURL("http://host.docker.internal", true); err != nil || root != "http://host.docker.internal" {
			t.Fatalf("root = %q, err = %v", root, err)
		}
	})
	t.Run("returns skills from the Skill Hub marketplace", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `{"plugins":[{"name":"terraform","description":"Terraform conventions","enabled":true}]}`})
		skills := mustList(t, server.URL)
		if len(skills) != 1 || skills[0].Name != "terraform" || skills[0].Enabled == nil || !*skills[0].Enabled {
			t.Fatalf("skills = %+v", skills)
		}
		r := seen()[0]
		if r.Path != "/claude-code/marketplace.json" || r.Accept != "application/json" || r.Auth != "Bearer sk-test" {
			t.Fatalf("request = %+v", r)
		}
	})
	t.Run("does not query the legacy endpoint when the marketplace is available or empty", func(t *testing.T) {
		for _, body := range []string{`{"plugins":[{"name":"hub","description":"Hub guidance"}]}`, `{"plugins":[]}`} {
			server, seen := skillsServer(t, scriptedReply{200, body})
			mustList(t, server.URL)
			if len(seen()) != 1 {
				t.Fatalf("%s: requests = %v", body, paths(seen()))
			}
		}
	})
	t.Run("falls back to the Skills Gateway list endpoint", func(t *testing.T) {
		for name, first := range map[string]scriptedReply{
			"404":        {404, `{}`},
			"server":     {500, `{"error":"skill hub unavailable"}`},
			"malformed":  {200, `{`},
			"connection": {-1, ``},
		} {
			server, seen := skillsServer(t, first, scriptedReply{200, `{"data":[{"id":"skill-1","name":"legacy"}]}`})
			if got := names(mustList(t, server.URL)); !reflect.DeepEqual(got, []string{"legacy"}) {
				t.Fatalf("%s: got %v", name, got)
			}
			if got := paths(seen()); !reflect.DeepEqual(got, []string{"GET /claude-code/marketplace.json", "GET /v1/skills"}) {
				t.Fatalf("%s: requests = %v", name, got)
			}
		}
	})
	t.Run("returns skills from the Skills Gateway as a bare array", func(t *testing.T) {
		server, _ := skillsServer(t, scriptedReply{200, `[{"id":"skill-1","name":"terraform","enabled":true}]`})
		if got := names(mustList(t, server.URL)); !reflect.DeepEqual(got, []string{"terraform"}) {
			t.Fatalf("got %v", got)
		}
	})
	t.Run("uses a short in-memory cache for repeated reads", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `[{"name":"terraform"}]`})
		mustList(t, server.URL)
		mustList(t, server.URL)
		if len(seen()) != 1 {
			t.Fatalf("requests = %v", paths(seen()))
		}
	})
}

func TestSkillHelpers(t *testing.T) {
	source := map[string]any{"type": "git", "url": "https://github.com/acme/skills.git"}
	t.Run("rejects skills without Skill Hub source metadata", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `{}`})
		_, err := createSkill(context.Background(), server.URL, "sk-test", skillInput{Name: "terraform"}, nil, false)
		if err == nil || !strings.Contains(err.Error(), "source is required") || len(seen()) != 0 {
			t.Fatalf("err = %v, requests = %v", err, paths(seen()))
		}
	})
	t.Run("creates skills through the Skill Hub plugins endpoint", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `{"name":"terraform"}`}, scriptedReply{204, ``})
		result, err := createSkill(context.Background(), server.URL, "sk-test", skillInput{Name: "terraform", Description: "Terraform conventions", Source: source}, nil, false)
		if err != nil || !reflect.DeepEqual(result, map[string]any{"name": "terraform"}) {
			t.Fatalf("result = %v, err = %v", result, err)
		}
		if got := paths(seen()); !reflect.DeepEqual(got, []string{"POST /claude-code/plugins", "POST /claude-code/plugins/terraform/enable"}) {
			t.Fatalf("requests = %v", got)
		}
	})
	t.Run("does not fall back to the Anthropic Skills API when the Skill Hub is missing", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{404, ``})
		_, err := createSkill(context.Background(), server.URL, "sk-test", skillInput{Name: "terraform", Source: source}, nil, false)
		if err == nil || err.Error() != "LiteLLM skill create failed: HTTP 404" || len(seen()) != 1 {
			t.Fatalf("err = %v, requests = %v", err, paths(seen()))
		}
	})
	deleteCases := []struct {
		name     string
		replies  []scriptedReply
		wantErr  string
		wantReqs []string
	}{
		{"deletes skills by id", []scriptedReply{{204, ``}}, "", []string{"DELETE /claude-code/plugins/skill-1"}},
		{"falls back on 404", []scriptedReply{{404, `{}`}, {204, ``}}, "", []string{"DELETE /claude-code/plugins/skill-1", "DELETE /v1/skills/skill-1"}},
		{"falls back on server error", []scriptedReply{{500, `{}`}, {204, ``}}, "", []string{"DELETE /claude-code/plugins/skill-1", "DELETE /v1/skills/skill-1"}},
		{"falls back on request failure", []scriptedReply{{-1, ``}, {204, ``}}, "", []string{"DELETE /claude-code/plugins/skill-1", "DELETE /v1/skills/skill-1"}},
		{"reports the Skill Hub failure when the legacy delete finds nothing", []scriptedReply{{500, `{}`}, {404, `{}`}}, "LiteLLM skill delete failed: HTTP 500", []string{"DELETE /claude-code/plugins/skill-1", "DELETE /v1/skills/skill-1"}},
	}
	for _, c := range deleteCases {
		t.Run(c.name, func(t *testing.T) {
			server, seen := skillsServer(t, c.replies...)
			err := deleteSkill(context.Background(), server.URL, "sk-test", "skill-1", nil, false)
			if c.wantErr == "" && err != nil || c.wantErr != "" && (err == nil || err.Error() != c.wantErr) {
				t.Fatalf("err = %v, want %q", err, c.wantErr)
			}
			if got := paths(seen()); !reflect.DeepEqual(got, c.wantReqs) {
				t.Fatalf("requests = %v", got)
			}
		})
	}
	t.Run("formats enabled skills as a system-prompt section", func(t *testing.T) {
		yes, no := true, false
		section := createSkillsPromptSection([]types.LiteLLMSkill{
			{ID: "enabled", Name: "terraform", Description: "Terraform conventions", Enabled: &yes},
			{ID: "disabled", Name: "legacy", Description: "Old guidance", Enabled: &no},
			{Name: `a"<b>&`},
		})
		for _, want := range []string{"<litellm_skills>", `<skill name="terraform">`, "Terraform conventions", `<skill name="a&quot;&lt;b&gt;&amp;">`, "No description provided."} {
			if !strings.Contains(section, want) {
				t.Errorf("missing %q in %s", want, section)
			}
		}
		if strings.Contains(section, "Old guidance") {
			t.Error("disabled skill leaked")
		}
		if createSkillsPromptSection(nil) != "" {
			t.Error("empty list should yield no section")
		}
	})
}

func staticAuth(base string) func(sdk.Context) (types.LiteLLMRuntimeAuth, error) {
	return func(sdk.Context) (types.LiteLLMRuntimeAuth, error) {
		return types.LiteLLMRuntimeAuth{BaseURL: base, APIKey: "fresh-token", Headers: map[string]string{"x-tenant": "new"}}, nil
	}
}

func TestSkillToolDefinitions(t *testing.T) {
	definitions := skillToolDefinitions(staticAuth("https://litellm.example.com"))
	t.Run("creates tools for listing, creating, and deleting skills", func(t *testing.T) {
		var got []string
		for _, d := range definitions {
			got = append(got, d.Name)
		}
		if want := []string{"litellm_skill_list", "litellm_skill_create", "litellm_skill_delete"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v", got)
		}
	})
	t.Run("groups the tools under one namespace with truthful annotation hints", func(t *testing.T) {
		want := []string{
			`{"readOnlyHint":true,"openWorldHint":false}`,
			`{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}`,
			`{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":true,"openWorldHint":false}`,
		}
		for i, d := range definitions {
			if d.Namespace == nil || d.Namespace.Name != "litellm_skills" {
				t.Errorf("%s namespace = %+v", d.Name, d.Namespace)
			}
			if got, _ := json.Marshal(d.Annotations); string(got) != want[i] {
				t.Errorf("%s annotations = %s", d.Name, got)
			}
		}
	})
	t.Run("executes the list tool with a fresh token", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `[{"id":"skill-1","name":"terraform","description":"Terraform conventions"}]`})
		result, err := skillToolDefinitions(staticAuth(server.URL))[0].Execute(sdk.Context{}, map[string]any{})
		if err != nil {
			t.Fatal(err)
		}
		if text := result.(sdk.ToolResult).Content; !strings.Contains(text, "terraform") {
			t.Fatalf("content = %q", text)
		}
		r := seen()[0]
		if r.Path != "/claude-code/marketplace.json" || r.Auth != "Bearer fresh-token" || r.Tenant != "new" {
			t.Fatalf("request = %+v", r)
		}
	})
	t.Run("executes the create tool with Skill Hub source metadata", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `{"name":"terraform"}`})
		result, err := skillToolDefinitions(staticAuth(server.URL))[1].Execute(sdk.Context{}, map[string]any{
			"name": "terraform", "description": "Terraform conventions",
			"sourceJson": `{"type":"git","url":"https://github.com/acme/skills.git"}`,
		})
		if err != nil || result.(sdk.ToolResult).Content != "LiteLLM skill created." {
			t.Fatalf("result = %v, err = %v", result, err)
		}
		var body map[string]any
		_ = json.Unmarshal([]byte(seen()[0].Body), &body)
		want := map[string]any{"name": "terraform", "description": "Terraform conventions", "source": map[string]any{"type": "git", "url": "https://github.com/acme/skills.git"}}
		if !reflect.DeepEqual(body, want) {
			t.Fatalf("body = %v", body)
		}
	})
	t.Run("rejects a sourceJson that is not an object", func(t *testing.T) {
		_, err := definitions[1].Execute(sdk.Context{}, map[string]any{"name": "x", "sourceJson": "[]"})
		if err == nil || err.Error() != "sourceJson must be a JSON object" {
			t.Fatalf("err = %v", err)
		}
	})
}

func skillsHookState() *extensionState {
	return newExtensionState(nil, []providerDefinition{{Name: providerName}}, nil)
}

func TestBeforeAgentStartSkillsHook(t *testing.T) {
	var diagnostics []string
	previous := reportDiagnostic
	reportDiagnostic = func(message string) { diagnostics = append(diagnostics, message) }
	t.Cleanup(func() { reportDiagnostic = previous })

	t.Run("survives an unrefreshable OAuth credential", func(t *testing.T) {
		diagnostics = nil
		got := skillsHookState().skillsSystemPrompt(context.Background(), "Base prompt", func() (*types.LiteLLMRuntimeAuth, error) {
			return nil, errors.New("OAuth refresh failed for litellm: LiteLLM credential cannot be refreshed")
		})
		if got != nil || len(diagnostics) != 0 {
			t.Fatalf("got %v, diagnostics %v", got, diagnostics)
		}
	})
	t.Run("survives a failing skills catalog request", func(t *testing.T) {
		server, _ := skillsServer(t, scriptedReply{-1, ``})
		got := skillsHookState().skillsSystemPrompt(context.Background(), "Base prompt", func() (*types.LiteLLMRuntimeAuth, error) {
			return &types.LiteLLMRuntimeAuth{BaseURL: server.URL, APIKey: "sk-test"}, nil
		})
		if got != nil {
			t.Fatalf("got %v", got)
		}
	})
	t.Run("rejects placeholder runtime hosts before sending credentials", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `[]`})
		_ = server
		got := skillsHookState().skillsSystemPrompt(context.Background(), "Base prompt", func() (*types.LiteLLMRuntimeAuth, error) {
			root, err := requireCredentialRoot("https://litellm.example.com", providerName)
			if err != nil {
				return nil, err
			}
			return &types.LiteLLMRuntimeAuth{BaseURL: root, APIKey: "sk-test"}, nil
		})
		if got != nil || len(seen()) != 0 {
			t.Fatalf("got %v, requests %v", got, paths(seen()))
		}
	})
	t.Run("reports the reason on stderr under LITELLM_VERBOSE_DISCOVERY", func(t *testing.T) {
		t.Setenv(envVerboseDiscovery, "1")
		diagnostics = nil
		skillsHookState().skillsSystemPrompt(context.Background(), "Base prompt", func() (*types.LiteLLMRuntimeAuth, error) {
			return nil, errors.New("LiteLLM credential cannot be refreshed")
		})
		if len(diagnostics) != 1 || !strings.Contains(diagnostics[0], "cannot be refreshed") {
			t.Fatalf("diagnostics = %v", diagnostics)
		}
	})
	t.Run("appends the skills section to the system prompt", func(t *testing.T) {
		server, _ := skillsServer(t, scriptedReply{200, `{"plugins":[{"name":"terraform","description":"Terraform conventions"}]}`})
		state := skillsHookState()
		got, _ := state.skillsSystemPrompt(context.Background(), "Base prompt", func() (*types.LiteLLMRuntimeAuth, error) {
			return &types.LiteLLMRuntimeAuth{BaseURL: server.URL, APIKey: "sk-test"}, nil
		}).(map[string]any)
		prompt, _ := got["systemPrompt"].(string)
		if !strings.HasPrefix(prompt, "Base prompt\n\nLiteLLM Skills Gateway") || state.defaultRuntimeAuth == nil {
			t.Fatalf("prompt = %q", prompt)
		}
	})
	t.Run("skips skills when discovery is disabled", func(t *testing.T) {
		t.Setenv(envOffline, "1")
		got := skillsHookState().skillsSystemPrompt(context.Background(), "Base prompt", func() (*types.LiteLLMRuntimeAuth, error) {
			t.Fatal("auth must not be resolved")
			return nil, nil
		})
		if got != nil {
			t.Fatalf("got %v", got)
		}
	})
}

func TestSkillsFromBodyKeepsTheWellFormedElements(t *testing.T) {
	server, _ := skillsServer(t, scriptedReply{200, `{"plugins":[{"name":"a"},"junk",{"name":5},{"name":"b","description":"d"}]}`})
	if got := names(mustList(t, server.URL)); !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("names = %v, want [a b]", got)
	}
}

func TestSkillsRequestsFollowTheHandlersContext(t *testing.T) {
	server, seen := skillsServer(t, scriptedReply{200, `[{"name":"a"}]`})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	skills, err := listSkills(ctx, server.URL, "sk-test", nil, false)
	if !errors.Is(err, context.Canceled) || len(skills) != 0 || len(seen()) != 0 {
		t.Fatalf("a cancelled turn still fetched: skills %v, err %v, requests %v", skills, err, paths(seen()))
	}
	// The failed attempt must not be cached as an empty list for the next turn.
	if got := names(mustList(t, server.URL)); !reflect.DeepEqual(got, []string{"a"}) {
		t.Fatalf("names after the cancelled turn = %v", got)
	}
}

func TestSkillsThroughTheHooks(t *testing.T) {
	t.Run("disables LiteLLM skills through settings", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `{"data":[]}`})
		hooks, _, host := wire(t, `{"litellm":{"skills":{"enabled":false}}}`)
		host.setAuth(providerName, &types.LiteLLMRuntimeAuth{BaseURL: server.URL, APIKey: "sk-test"})
		if slices.Contains(hooks.tools, "litellm_skill_list") || len(hooks.tools) != 0 {
			t.Fatalf("tools = %v", hooks.tools)
		}
		for _, result := range hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": "Base prompt"}) {
			if result != nil {
				t.Fatalf("a before_agent_start handler answered: %v", result)
			}
		}
		if len(seen()) != 0 {
			t.Fatalf("requests = %v", paths(seen()))
		}
	})

	t.Run("clears cached Skills auth when the host reports revoked credentials", func(t *testing.T) {
		server, seen := skillsServer(t, scriptedReply{200, `[]`})
		hooks, state, host := wire(t, "")
		host.setAuth(providerName, &types.LiteLLMRuntimeAuth{BaseURL: server.URL, APIKey: "active-key"})
		data := map[string]any{"systemPrompt": "Base prompt"}
		hooks.emit(t, sdk.EventBeforeAgentStart, host, data)
		if state.defaultRuntimeAuth == nil || len(seen()) == 0 {
			t.Fatalf("auth not cached: %v, requests %v", state.defaultRuntimeAuth, paths(seen()))
		}
		before := len(seen())
		host.setAuth(providerName, nil)
		hooks.emit(t, sdk.EventBeforeAgentStart, host, data)
		if state.defaultRuntimeAuth != nil {
			t.Fatalf("auth kept after revocation: %v", state.defaultRuntimeAuth)
		}
		if _, err := state.resolveDefaultRuntimeAuth(nil); err == nil || !strings.Contains(err.Error(), "no credentials for litellm") {
			t.Fatalf("err = %v", err)
		}
		if len(seen()) != before {
			t.Fatalf("requests after revocation: %v", paths(seen()))
		}
	})
}
