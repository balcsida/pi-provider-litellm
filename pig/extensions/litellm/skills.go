package litellm

// Ports src/skills.ts and the skills part of the src/index.ts factory: the Skills Gateway client, the
// litellm_skill_* tools and the before_agent_start hook that appends the active skills to the system prompt.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const (
	skillsCacheTTL     = 60 * time.Second
	skillsRequestLimit = 10 * time.Second
)

type skillsCacheEntry struct {
	baseURL, apiKey string
	fetchedAt       time.Time
	skills          []types.LiteLLMSkill
}

var (
	skillsCacheMu sync.Mutex
	skillsCache   *skillsCacheEntry
)

// resetSkillsCache is resetSkillsCache.
func resetSkillsCache() {
	skillsCacheMu.Lock()
	skillsCache = nil
	skillsCacheMu.Unlock()
}

// skillsFromBody is getSkillsFromBody: a bare array, or the array under plugins, data or skills.
func skillsFromBody(body any) []types.LiteLLMSkill {
	list, _ := body.([]any)
	if record, ok := body.(map[string]any); ok {
		for _, key := range []string{"plugins", "data", "skills"} {
			if array, ok := record[key].([]any); ok {
				list = array
				break
			}
		}
	}
	// Decoded per element: one malformed entry must not discard the rest of the list.
	var skills []types.LiteLLMSkill
	for _, element := range list {
		raw, _ := json.Marshal(element)
		var skill types.LiteLLMSkill
		if json.Unmarshal(raw, &skill) == nil {
			skills = append(skills, skill)
		}
	}
	return skills
}

// skillsRequest sends one bounded request with the credential and custom headers.
func skillsRequest(parent context.Context, method, target, apiKey string, headers map[string]string, accept string, body []byte) (*http.Response, error) {
	ctx, cancel := context.WithTimeout(parent, skillsRequestLimit)
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequestWithContext(ctx, method, target, reader)
	if err != nil {
		cancel()
		return nil, err
	}
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	request.Header.Set("Authorization", "Bearer "+apiKey)
	if accept != "" {
		request.Header.Set("Accept", accept)
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		cancel()
		return nil, err
	}
	response.Body = cancelOnClose{response.Body, cancel}
	return response, nil
}

type cancelOnClose struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (c cancelOnClose) Close() error {
	defer c.cancel()
	return c.ReadCloser.Close()
}

// listSkills is listSkills: the Skill Hub marketplace first, then /v1/skills; failures yield no skills.
func listSkills(ctx context.Context, baseURL, apiKey string, headers map[string]string, allowInsecureHTTP bool) ([]types.LiteLLMSkill, error) {
	root, err := protocols.NormalizeBaseURL(baseURL, allowInsecureHTTP)
	if err != nil {
		return nil, err
	}
	skillsCacheMu.Lock()
	if c := skillsCache; c != nil && c.baseURL == root && c.apiKey == apiKey && time.Since(c.fetchedAt) < skillsCacheTTL {
		skills := c.skills
		skillsCacheMu.Unlock()
		return skills, nil
	}
	skillsCacheMu.Unlock()

	fetch := func(path string) ([]types.LiteLLMSkill, bool) {
		response, err := skillsRequest(ctx, http.MethodGet, root+path, apiKey, headers, "application/json", nil)
		if err != nil {
			return nil, false
		}
		defer response.Body.Close()
		if response.StatusCode < 200 || response.StatusCode > 299 {
			return nil, false
		}
		var body any
		if json.NewDecoder(response.Body).Decode(&body) != nil {
			return nil, false
		}
		return skillsFromBody(body), true
	}
	skills, ok := fetch("/claude-code/marketplace.json")
	if !ok {
		skills, _ = fetch("/v1/skills")
	}
	// A cancelled turn says nothing about the proxy: do not cache its empty answer for the next one.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	skillsCacheMu.Lock()
	skillsCache = &skillsCacheEntry{baseURL: root, apiKey: apiKey, fetchedAt: time.Now(), skills: skills}
	skillsCacheMu.Unlock()
	return skills, nil
}

type skillInput struct {
	Name        string
	Description string
	Source      map[string]any
}

// createSkill is createSkill. Only the Skill Hub creates skills from JSON: LiteLLM's POST /v1/skills is
// Anthropic's multipart Skills API, so a JSON code skill sent there never becomes a listable skill.
func createSkill(ctx context.Context, baseURL, apiKey string, input skillInput, headers map[string]string, allowInsecureHTTP bool) (any, error) {
	if input.Source == nil {
		return nil, errors.New("source is required: skills are created through the LiteLLM Skill Hub")
	}
	root, err := protocols.NormalizeBaseURL(baseURL, allowInsecureHTTP)
	if err != nil {
		return nil, err
	}
	payload := map[string]any{"name": input.Name, "source": input.Source}
	if input.Description != "" {
		payload["description"] = input.Description
	}
	body, _ := json.Marshal(payload)
	response, err := skillsRequest(ctx, http.MethodPost, root+"/claude-code/plugins", apiKey, headers, "", body)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode > 299 {
		return nil, fmt.Errorf("LiteLLM skill create failed: HTTP %d", response.StatusCode)
	}
	resetSkillsCache()
	enable, err := skillsRequest(ctx, http.MethodPost, root+"/claude-code/plugins/"+url.PathEscape(input.Name)+"/enable", apiKey, headers, "", nil)
	if err != nil {
		return nil, err
	}
	enable.Body.Close()
	if enable.StatusCode < 200 || enable.StatusCode > 299 {
		return nil, fmt.Errorf("LiteLLM skill enable failed: HTTP %d", enable.StatusCode)
	}
	var result any
	if json.NewDecoder(response.Body).Decode(&result) != nil {
		return map[string]any{}, nil
	}
	return result, nil
}

// deleteSkill is deleteSkill: the Skill Hub first, the Skills Gateway when it is missing or failing.
func deleteSkill(ctx context.Context, baseURL, apiKey, skillID string, headers map[string]string, allowInsecureHTTP bool) error {
	root, err := protocols.NormalizeBaseURL(baseURL, allowInsecureHTTP)
	if err != nil {
		return err
	}
	hubStatus := 0
	response, hubErr := skillsRequest(ctx, http.MethodDelete, root+"/claude-code/plugins/"+url.PathEscape(skillID), apiKey, headers, "application/json", nil)
	if hubErr == nil {
		response.Body.Close()
		hubStatus = response.StatusCode
	}
	status := hubStatus
	if hubErr != nil || hubStatus == http.StatusNotFound || hubStatus >= 500 {
		legacy, err := skillsRequest(ctx, http.MethodDelete, root+"/v1/skills/"+url.PathEscape(skillID), apiKey, headers, "", nil)
		if err != nil {
			return err
		}
		legacy.Body.Close()
		status = legacy.StatusCode
	}
	if status == http.StatusNotFound && hubStatus != http.StatusNotFound {
		if hubErr != nil {
			return hubErr
		}
		return fmt.Errorf("LiteLLM skill delete failed: HTTP %d", hubStatus)
	}
	if (status < 200 || status > 299) && status != http.StatusNotFound {
		return fmt.Errorf("LiteLLM skill delete failed: HTTP %d", status)
	}
	resetSkillsCache()
	return nil
}

var xmlEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;")

// createSkillsPromptSection is createSkillsPromptSection; "" is undefined.
func createSkillsPromptSection(skills []types.LiteLLMSkill) string {
	var entries []string
	for _, skill := range skills {
		if (skill.Enabled != nil && !*skill.Enabled) || skill.Name == "" {
			continue
		}
		description := strings.TrimSpace(skill.Description)
		if description == "" {
			description = "No description provided."
		}
		entries = append(entries, `<skill name="`+xmlEscaper.Replace(skill.Name)+"\">\n"+xmlEscaper.Replace(description)+"\n</skill>")
	}
	if len(entries) == 0 {
		return ""
	}
	return "LiteLLM Skills Gateway provided these active skills. Use them as additional guidance when relevant.\n<litellm_skills>\n" +
		strings.Join(entries, "\n") + "\n</litellm_skills>"
}

func formatSkills(skills []types.LiteLLMSkill) string {
	if len(skills) == 0 {
		return "No LiteLLM skills are registered."
	}
	lines := make([]string, len(skills))
	for i, skill := range skills {
		disabled := ""
		if skill.Enabled != nil && !*skill.Enabled {
			disabled = " (disabled)"
		}
		lines[i] = fmt.Sprintf("- %s%s: %s", skill.Name, disabled, skill.Description)
	}
	return strings.Join(lines, "\n")
}

func parseJSONObject(value, fieldName string) (map[string]any, error) {
	var parsed any
	if err := json.Unmarshal([]byte(value), &parsed); err != nil {
		return nil, err
	}
	object, ok := parsed.(map[string]any)
	if !ok || object == nil {
		return nil, fmt.Errorf("%s must be a JSON object", fieldName)
	}
	return object, nil
}

func boolPtr(value bool) *bool { return &value }

func stringParam(params map[string]any, name string) (string, error) {
	value, ok := params[name].(string)
	if !ok {
		return "", fmt.Errorf("%s must be a string", name)
	}
	return value, nil
}

// skillToolDefinitions is createSkillToolDefinitions. Pi groups tools by namespace and lets permission
// extensions read annotation hints.
func skillToolDefinitions(getAuth func(sdk.Context) (types.LiteLLMRuntimeAuth, error)) []sdk.ToolDefinition {
	namespace := &sdk.ToolNamespace{Name: "litellm_skills", Description: "LiteLLM Skills Gateway"}
	return []sdk.ToolDefinition{
		{
			Name:          "litellm_skill_list",
			Label:         "LiteLLM Skills",
			Description:   "List skills registered on the LiteLLM proxy Skills Gateway.",
			PromptSnippet: "List LiteLLM Skills Gateway skills",
			Namespace:     namespace,
			Annotations:   &sdk.ToolAnnotations{ReadOnlyHint: boolPtr(true), OpenWorldHint: boolPtr(false)},
			Parameters:    sdk.Schema{"type": "object", "properties": map[string]any{}},
			Execute: func(ctx sdk.Context, _ map[string]any) (any, error) {
				auth, err := getAuth(ctx)
				if err != nil {
					return nil, err
				}
				skills, err := listSkills(requestContext{ctx}, auth.BaseURL, auth.APIKey, auth.Headers, auth.AllowInsecureHTTP)
				if err != nil {
					return nil, err
				}
				return sdk.ToolResult{Content: formatSkills(skills), Details: map[string]any{"count": len(skills)}}, nil
			},
		},
		{
			Name:        "litellm_skill_create",
			Label:       "Create LiteLLM Skill",
			Description: "Create and enable a skill on the LiteLLM proxy Skill Hub from source metadata.",
			Namespace:   namespace,
			Annotations: &sdk.ToolAnnotations{ReadOnlyHint: boolPtr(false), DestructiveHint: boolPtr(false), IdempotentHint: boolPtr(false), OpenWorldHint: boolPtr(false)},
			Parameters: sdk.Schema{
				"type": "object",
				"properties": map[string]any{
					"name":        map[string]any{"type": "string", "description": "Skill name"},
					"description": map[string]any{"type": "string", "description": "Skill description"},
					"sourceJson":  map[string]any{"type": "string", "description": "Skill Hub source metadata JSON object"},
				},
				"required": []string{"name", "sourceJson"},
			},
			Execute: func(ctx sdk.Context, params map[string]any) (any, error) {
				auth, err := getAuth(ctx)
				if err != nil {
					return nil, err
				}
				name, err := stringParam(params, "name")
				if err != nil {
					return nil, err
				}
				sourceJSON, err := stringParam(params, "sourceJson")
				if err != nil {
					return nil, err
				}
				description, _ := params["description"].(string)
				source, err := parseJSONObject(sourceJSON, "sourceJson")
				if err != nil {
					return nil, err
				}
				result, err := createSkill(requestContext{ctx}, auth.BaseURL, auth.APIKey, skillInput{Name: name, Description: description, Source: source}, auth.Headers, auth.AllowInsecureHTTP)
				if err != nil {
					return nil, err
				}
				return sdk.ToolResult{Content: "LiteLLM skill created.", Details: map[string]any{"result": result}}, nil
			},
		},
		{
			Name:        "litellm_skill_delete",
			Label:       "Delete LiteLLM Skill",
			Description: "Delete a skill from the LiteLLM proxy Skills Gateway.",
			Namespace:   namespace,
			Annotations: &sdk.ToolAnnotations{ReadOnlyHint: boolPtr(false), DestructiveHint: boolPtr(true), IdempotentHint: boolPtr(true), OpenWorldHint: boolPtr(false)},
			Parameters: sdk.Schema{
				"type":       "object",
				"properties": map[string]any{"skillId": map[string]any{"type": "string", "description": "LiteLLM skill id"}},
				"required":   []string{"skillId"},
			},
			Execute: func(ctx sdk.Context, params map[string]any) (any, error) {
				auth, err := getAuth(ctx)
				if err != nil {
					return nil, err
				}
				skillID, err := stringParam(params, "skillId")
				if err != nil {
					return nil, err
				}
				if err := deleteSkill(requestContext{ctx}, auth.BaseURL, auth.APIKey, skillID, auth.Headers, auth.AllowInsecureHTTP); err != nil {
					return nil, err
				}
				return sdk.ToolResult{Content: "LiteLLM skill deleted: " + skillID, Details: map[string]any{}}, nil
			},
		},
	}
}

// skillsSystemPrompt is the before_agent_start body. Skills enrichment is best-effort: an expired
// credential or an unreachable proxy must not report an extension error on every turn, so errors only
// produce a verbose diagnostic. It returns the replacement prompt, or nil to keep it.
func (s *extensionState) skillsSystemPrompt(ctx context.Context, systemPrompt string, getAuth func() (*types.LiteLLMRuntimeAuth, error)) any {
	s.mu.Lock()
	s.defaultRuntimeAuth = nil
	s.mu.Unlock()
	if discoveryDisabledReason() != "" {
		return nil
	}
	skip := func(err error) any {
		if isVerboseDiscovery() {
			reportDiagnostic("LiteLLM: skipping Skills (" + err.Error() + ").")
		}
		return nil
	}
	auth, err := getAuth()
	if err != nil {
		return skip(err)
	}
	if auth == nil {
		return nil
	}
	s.mu.Lock()
	s.defaultRuntimeAuth = auth
	s.mu.Unlock()
	skills, err := listSkills(ctx, auth.BaseURL, auth.APIKey, auth.Headers, auth.AllowInsecureHTTP)
	if err != nil {
		return skip(err)
	}
	section := createSkillsPromptSection(skills)
	if section == "" {
		return nil
	}
	return map[string]any{"systemPrompt": systemPrompt + "\n\n" + section}
}

// setupSkills registers the skill tools and the prompt hook when the "skills" feature is enabled.
func setupSkills(e hookRegistrar, s *extensionState) {
	if !isFeatureEnabled(s.settings, "skills") {
		return
	}
	for _, definition := range skillToolDefinitions(func(ctx sdk.Context) (types.LiteLLMRuntimeAuth, error) {
		return s.resolveDefaultRuntimeAuth(&ctx)
	}) {
		e.RegisterTool(definition)
	}
	e.OnEvent(sdk.EventBeforeAgentStart, func(h hookHost, data map[string]any) (any, error) {
		prompt, _ := data["systemPrompt"].(string)
		return s.skillsSystemPrompt(h.RequestContext(), prompt, func() (*types.LiteLLMRuntimeAuth, error) {
			return h.ResolveAuth(s.definitions[0].Name)
		}), nil
	})
}
