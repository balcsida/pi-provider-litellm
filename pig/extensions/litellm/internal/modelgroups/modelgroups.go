// Ports src/model-groups.ts
package modelgroups

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/backend"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/thinking"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const builtinContextWindow = 128_000

// DefaultMaxTokens is the output limit used when neither /model/info nor the catalog reports one.
const DefaultMaxTokens = 16_384

const maxSafeInteger = 1<<53 - 1

// DefaultContextWindow is the window used when neither /model/info nor the catalog reports one.
// LITELLM_DEFAULT_CONTEXT_WINDOW raises it for proxies whose model map cannot populate
// `max_input_tokens`; an unset or unusable value keeps the 128K default.
func DefaultContextWindow() int {
	// A whole-string numeric parse, like Number(): a numeric prefix such as `922000junk` or a
	// fraction such as `1.5` is rejected instead of passing as a limit.
	parsed, err := strconv.ParseFloat(strings.TrimSpace(os.Getenv("LITELLM_DEFAULT_CONTEXT_WINDOW")), 64)
	if err != nil || parsed != math.Trunc(parsed) || parsed <= 0 || parsed > maxSafeInteger {
		return builtinContextWindow
	}
	return int(parsed)
}

// IsFallbackContextWindow reports whether a stored window is one this extension assumed rather
// than read. A model cached under the built-in default must still read as evidence-free once an
// operator configures LITELLM_DEFAULT_CONTEXT_WINDOW.
// ponytail: a window stored under a setting that was later lowered is indistinguishable from
// measured evidence and stays until the next online discovery.
func IsFallbackContextWindow(contextWindow float64) bool {
	return contextWindow == builtinContextWindow || contextWindow == float64(DefaultContextWindow())
}

// SemanticFamily is the backend family a catalog identified.
type SemanticFamily string

const (
	SemanticFamilyClaude   SemanticFamily = "claude"
	SemanticFamilyDeepSeek SemanticFamily = "deepseek"
	SemanticFamilyGemini   SemanticFamily = "gemini"
	SemanticFamilyKimi     SemanticFamily = "kimi"
	SemanticFamilyOpenAI   SemanticFamily = "openai"
)

// FamilyEvidence is a SemanticFamily or FamilyConflicting; the empty string is absent.
type FamilyEvidence = SemanticFamily

// FamilyConflicting is the "conflicting" family evidence.
const FamilyConflicting FamilyEvidence = "conflicting"

// SemanticModel names a model generation with its own reasoning contract; empty is absent.
type SemanticModel string

const (
	SemanticModelDeepSeekV4  SemanticModel = "deepseek-v4"
	SemanticModelKimiK25K26  SemanticModel = "kimi-k2.5-k2.6"
	SemanticModelKimiK27Code SemanticModel = "kimi-k2.7-code"
	SemanticModelKimiK3      SemanticModel = "kimi-k3"
)

const (
	reasoningEffortParam      = "reasoning_effort"
	thinkingParam             = "thinking"
	messagesEndpoint          = "/v1/messages"
	jsonDepthLimitPlaceholder = "[depth-limit]"
)

// MessagesBackendCompat is the Pick of forceAdaptiveThinking and supportsTemperature.
type MessagesBackendCompat struct {
	ForceAdaptiveThinking *bool `json:"forceAdaptiveThinking,omitempty"`
	SupportsTemperature   *bool `json:"supportsTemperature,omitempty"`
}

// ReasoningCompat is the Pick of the three compat fields a reasoning policy states.
type ReasoningCompat struct {
	RequiresReasoningContentOnAssistantMessages *bool
	SupportsReasoningEffort                     *bool
	ThinkingFormat                              string
}

// ReasoningPolicy: ThinkingLevelMap nil is absent, a nil map value is a denied level.
type ReasoningPolicy struct {
	Reasoning        bool
	ThinkingLevelMap ai.ThinkingLevelMap
	Compat           *ReasoningCompat
}

// CatalogCost is the Partial cost a catalog resolution may state; nil fields are absent.
type CatalogCost struct {
	Input, Output, CacheRead, CacheWrite *float64
	Tiers                                []ai.CostTier
}

// CatalogResolution: nil pointers and empty strings are TypeScript `undefined`.
type CatalogResolution struct {
	Provider       string
	CatalogModelID string
	SemanticFamily FamilyEvidence
	SemanticModel  SemanticModel
	MessagesCompat *MessagesBackendCompat
	// MessagesStrictTools is evidence for the actual Messages route, kept separate from
	// model-generation serializer policy so strict-tool disagreement cannot change the transport.
	MessagesStrictTools      bool
	MessagesThinkingLevelMap ai.ThinkingLevelMap
	Reasoning                *bool
	EffortLevels             []string
	ThinkingLevelMap         ai.ThinkingLevelMap
	Vision                   *bool
	ContextWindow            *float64
	MaxTokens                *float64
	Cost                     *CatalogCost
}

// CatalogResolver resolves catalog evidence for one deployment; nil means none.
type CatalogResolver func(entry types.ModelInfoEntry) *CatalogResolution

// ReducedModelGroup is one public route reduced over all of its deployments.
type ReducedModelGroup struct {
	ID                               string
	API                              types.LiteLLMApi
	Reasoning                        bool
	AcceptsResponsesReasoningControl bool
	ThinkingLevelMap                 ai.ThinkingLevelMap
	Vision                           bool
	ContextWindow                    float64
	ContextWindowDefaulted           bool
	MaxTokens                        float64
	Cost                             ai.ModelCost
	HasCompleteCost                  bool
	HasCompleteMetadata              bool
	CatalogProvider                  string
	SemanticModel                    SemanticModel
	SemanticFamily                   FamilyEvidence
	MessagesCompat                   *MessagesBackendCompat
	MessagesStrictTools              bool
	// CatalogAuthorityAmbiguous is set when deployments disagreed on catalog provider identity, so
	// catalog limits, pricing, and reasoning metadata were withheld for the whole group.
	CatalogAuthorityAmbiguous bool
	// DeploymentFamilies has one entry per routable deployment. Empty means that deployment supplied
	// no usable family evidence; callers use this to gate outbound rewrites.
	DeploymentFamilies          []FamilyEvidence
	NormalizeThinkTags          bool
	SuppressReasoningVisibility bool
	AcceptedOpenAIParams        []string
	ReasoningPolicy             ReasoningPolicy
}

var (
	responsesModePattern = regexp.MustCompile(`(?i)^responses?$`)
	chatStyleModePattern = regexp.MustCompile(`(?i)^chat$`)
)

type costField int

const (
	costInput costField = iota
	costOutput
	costCacheRead
	costCacheWrite
	costFieldCount
)

// WireString returns value when it is a string (or a non-nil *string). `/model/info` is parsed JSON
// from operator-authored proxy config, so a declared string can arrive as another type; that
// withholds the row's evidence instead of failing.
func WireString(value any) (string, bool) {
	switch v := value.(type) {
	case string:
		return v, true
	case *string:
		if v != nil {
			return *v, true
		}
	}
	return "", false
}

// IsResponsesMode reports whether mode spells the Responses mode.
func IsResponsesMode(mode any) bool {
	value, ok := WireString(mode)
	return ok && responsesModePattern.MatchString(strings.TrimSpace(value))
}

// Mode is a normalized deployment mode.
type Mode string

const (
	ModeChat        Mode = "chat"
	ModeResponses   Mode = "responses"
	ModeUnknown     Mode = "unknown"
	ModeUnsupported Mode = "unsupported"
)

// NormalizedMode: a nil or unreadable mode remains in the conservative reduction as unknown instead
// of being filtered as unsupported and silently relaxing the remaining group.
func NormalizedMode(mode any) Mode {
	value, ok := WireString(mode)
	if !ok {
		return ModeUnknown
	}
	value = strings.TrimSpace(value)
	switch {
	case responsesModePattern.MatchString(value):
		return ModeResponses
	case chatStyleModePattern.MatchString(value):
		return ModeChat
	}
	return ModeUnsupported
}

// Canonicalization is depth-bounded because deployment metadata is untrusted.
const maxCanonicalDepth = 12

func sortValue(value any, depth int) any {
	// Do not traverse the remaining untrusted subtree at the cap.
	if depth >= maxCanonicalDepth {
		return jsonDepthLimitPlaceholder
	}
	switch v := value.(type) {
	case []any:
		out := make([]any, len(v))
		for i, child := range v {
			out[i] = sortValue(child, depth+1)
		}
		return out
	case map[string]any:
		// encoding/json writes map keys in sorted order.
		out := make(map[string]any, len(v))
		for key, child := range v {
			out[key] = sortValue(child, depth+1)
		}
		return out
	}
	return value
}

func marshalNoEscape(value any) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(value); err != nil {
		return ""
	}
	return strings.TrimSuffix(buf.String(), "\n")
}

func canonicalJSON(value any) (string, bool) {
	if value == nil {
		return "", false
	}
	if rv := reflect.ValueOf(value); rv.Kind() == reflect.Pointer && rv.IsNil() {
		return "", false
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return "", false
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var generic any
	if err := decoder.Decode(&generic); err != nil {
		return "", false
	}
	return marshalNoEscape(sortValue(generic, 0)), true
}

// StableJSON compares catalog policies without depending on key order. The second result is false
// for an absent (nil) value.
func StableJSON(value any) (string, bool) {
	return canonicalJSON(value)
}

func stableEntry(entry types.ModelInfoEntry) string {
	signature, _ := canonicalJSON(entry)
	// omitempty drops empty lists; keep absent and empty distinct as the wire does.
	info := entry.ModelInfo
	if info != nil {
		signature += "|" + strconv.FormatBool(info.SupportedOpenAIParams == nil) +
			strconv.FormatBool(info.SupportedEndpoints == nil) + strconv.FormatBool(info.ReasoningEffortLevels == nil)
	}
	if entry.LiteLLMParams != nil {
		signature += "|" + strconv.FormatBool(entry.LiteLLMParams.AllowedOpenAIParams == nil)
	}
	return signature
}

func entryModelInfoID(entry types.ModelInfoEntry) string {
	if entry.ModelInfo == nil {
		return ""
	}
	return strings.TrimSpace(entry.ModelInfo.ID)
}

// uniqueDeployments collapses exact identified duplicates, keeps conflicting variants sharing an id
// plural, and keeps anonymous rows distinct because no identity proves equality.
func uniqueDeployments(entries []types.ModelInfoEntry) []types.ModelInfoEntry {
	identified := map[string]map[string]types.ModelInfoEntry{}
	type anonymousRow struct {
		signature string
		entry     types.ModelInfoEntry
	}
	var anonymous []anonymousRow
	for _, entry := range entries {
		signature := stableEntry(entry)
		if id := entryModelInfoID(entry); id != "" {
			variants := identified[id]
			if variants == nil {
				variants = map[string]types.ModelInfoEntry{}
				identified[id] = variants
			}
			variants[signature] = entry
		} else {
			anonymous = append(anonymous, anonymousRow{signature, entry})
		}
	}
	ids := make([]string, 0, len(identified))
	for id := range identified {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var out []types.ModelInfoEntry
	for _, id := range ids {
		signatures := make([]string, 0, len(identified[id]))
		for signature := range identified[id] {
			signatures = append(signatures, signature)
		}
		sort.Strings(signatures)
		for _, signature := range signatures {
			out = append(out, identified[id][signature])
		}
	}
	sort.SliceStable(anonymous, func(i, j int) bool { return anonymous[i].signature < anonymous[j].signature })
	for _, row := range anonymous {
		out = append(out, row.entry)
	}
	return out
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// explicitLimit: a router limit is usable only when it is a finite positive token count. Without
// this, one deployment reporting 0 would clamp the whole group to 0.
func explicitLimit(value *float64) *float64 {
	if value == nil || !finite(*value) || *value <= 0 {
		return nil
	}
	return value
}

var extendedLevels = func() []ai.ThinkingLevel {
	levels := make([]ai.ThinkingLevel, len(thinking.Definitions))
	for i, definition := range thinking.Definitions {
		levels[i] = definition.Level
	}
	return levels
}()

func normalizeEffort(level string) (ai.ThinkingLevel, bool) {
	if level == "none" {
		level = "off"
	}
	for _, candidate := range extendedLevels {
		if string(candidate) == level {
			return candidate, true
		}
	}
	return "", false
}

// levelFlag reads the LiteLLM support flag for a level. It follows thinking.Definitions.
func levelFlag(info *types.ModelInfoDetails, level ai.ThinkingLevel) *bool {
	switch level {
	case ai.ThinkingOff:
		return info.SupportsNoneReasoningEffort
	case ai.ThinkingMinimal:
		return info.SupportsMinimalReasoningEffort
	case ai.ThinkingLow:
		return info.SupportsLowReasoningEffort
	case ai.ThinkingMedium:
		return info.SupportsMediumReasoningEffort
	case ai.ThinkingHigh:
		return info.SupportsHighReasoningEffort
	case ai.ThinkingXHigh:
		return info.SupportsXHighReasoningEffort
	case ai.ThinkingMax:
		return info.SupportsMaxReasoningEffort
	}
	return nil
}

func boolPtr(value bool) *bool { return &value }

func stringPtr(value string) *string { return &value }

// reportedLevel: LiteLLM reads a declared `reasoning_effort_levels` list whole, ahead of the
// per-level flags, because flags cannot state a set such as low/high/max: medium has no opt-out. A
// declared list therefore answers every level for its deployment. Nil is absent.
func reportedLevel(entry types.ModelInfoEntry, level ai.ThinkingLevel) *bool {
	info := entry.ModelInfo
	if info == nil {
		return nil
	}
	if info.ReasoningEffortLevels != nil {
		for _, effort := range info.ReasoningEffortLevels {
			if normalized, ok := normalizeEffort(effort); ok && normalized == level {
				return boolPtr(true)
			}
		}
		return boolPtr(false)
	}
	return levelFlag(info, level)
}

func isTrue(value *bool) bool  { return value != nil && *value }
func isFalse(value *bool) bool { return value != nil && !*value }

func levelWire(level ai.ThinkingLevel) *string {
	if level == ai.ThinkingOff {
		return stringPtr("none")
	}
	return stringPtr(string(level))
}

// reasoningLevelMap: a public effort list is complete: listed standard levels are enabled and
// omitted ones are denied. Without a public list, standard levels stay absent so Pi keeps its
// defaults. LiteLLM flags may add or remove levels, while xhigh/max always require an explicit
// LiteLLM opt-in.
func reasoningLevelMap(
	entries []types.ModelInfoEntry,
	catalogEfforts [][]string,
	catalogMaps []ai.ThinkingLevelMap,
) ai.ThinkingLevelMap {
	var publicSets []map[ai.ThinkingLevel]bool
	for _, levels := range catalogEfforts {
		if levels == nil {
			continue
		}
		set := map[ai.ThinkingLevel]bool{}
		for _, effort := range levels {
			if level, ok := normalizeEffort(effort); ok {
				set[level] = true
			}
		}
		if len(set) > 0 {
			publicSets = append(publicSets, set)
		}
	}
	out := ai.ThinkingLevelMap{}
	if len(publicSets) > 0 {
		for _, level := range []ai.ThinkingLevel{ai.ThinkingOff, ai.ThinkingMinimal, ai.ThinkingLow, ai.ThinkingMedium, ai.ThinkingHigh} {
			every := true
			for _, set := range publicSets {
				every = every && set[level]
			}
			if every {
				out[level] = levelWire(level)
			} else {
				out[level] = nil
			}
		}
	}
	// A catalog level map is not a complete list: a null denies, a value supplies the wire spelling,
	// and an omitted standard level keeps Pi's default.
	for _, catalogMap := range catalogMaps {
		for level, value := range catalogMap {
			existing, has := out[level]
			if value == nil || !(has && existing == nil) {
				out[level] = value
			}
		}
	}

	for _, level := range extendedLevels {
		allTrue := len(entries) > 0
		anyFalse := false
		for _, entry := range entries {
			reported := reportedLevel(entry, level)
			anyFalse = anyFalse || isFalse(reported)
			allTrue = allTrue && isTrue(reported)
		}
		if anyFalse {
			out[level] = nil
		} else if allTrue {
			out[level] = levelWire(level)
		}
	}
	for _, level := range []ai.ThinkingLevel{ai.ThinkingXHigh, ai.ThinkingMax} {
		every := true
		for _, entry := range entries {
			every = every && isTrue(reportedLevel(entry, level))
		}
		if !every {
			out[level] = nil
		}
	}
	return out
}

func deniedLevels(levels ai.ThinkingLevelMap) ai.ThinkingLevelMap {
	out := ai.ThinkingLevelMap{}
	for level, value := range levels {
		if value == nil {
			out[level] = nil
		}
	}
	return out
}

func normalizeParams(params []string) map[string]bool {
	out := map[string]bool{}
	for _, param := range params {
		if trimmed := strings.TrimSpace(param); trimmed != "" {
			out[trimmed] = true
		}
	}
	return out
}

func acceptedParams(entry types.ModelInfoEntry) map[string]bool {
	var supported, allowed []string
	if entry.ModelInfo != nil {
		supported = entry.ModelInfo.SupportedOpenAIParams
	}
	if entry.LiteLLMParams != nil {
		allowed = entry.LiteLLMParams.AllowedOpenAIParams
	}
	params := normalizeParams(supported)
	for param := range normalizeParams(allowed) {
		params[param] = true
	}
	return params
}

// optsIntoEffortCarrier: LiteLLM omits supported_openai_params (or returns null) for a deployment
// its model map does not describe, so that absence says nothing about the carrier. An explicit
// `supports_reasoning: true` is the operator's opt-in there.
func optsIntoEffortCarrier(entry types.ModelInfoEntry) bool {
	info := entry.ModelInfo
	return info != nil && info.SupportedOpenAIParams == nil && isTrue(info.SupportsReasoning)
}

func intersectParams(entries []types.ModelInfoEntry) []string {
	out := []string{}
	if len(entries) == 0 {
		return out
	}
	sets := make([]map[string]bool, len(entries))
	for i, entry := range entries {
		sets[i] = acceptedParams(entry)
	}
	for param := range sets[0] {
		shared := true
		for _, set := range sets[1:] {
			shared = shared && set[param]
		}
		if shared {
			out = append(out, param)
		}
	}
	sort.Strings(out)
	return out
}

// levelsOf builds a full level map in the order off, minimal, low, medium, high, xhigh, max; an
// empty string denies the level.
func levelsOf(off, minimal, low, medium, high, xhigh, max string) ai.ThinkingLevelMap {
	out := ai.ThinkingLevelMap{}
	for i, value := range []string{off, minimal, low, medium, high, xhigh, max} {
		if value == "" {
			out[extendedLevels[i]] = nil
		} else {
			out[extendedLevels[i]] = stringPtr(value)
		}
	}
	return out
}

func onlyHigh() ai.ThinkingLevelMap { return levelsOf("", "", "", "", "high", "", "") }

// NoTransmissibleLevels denies every level. pi-ai reads an ABSENT thinkingLevelMap as "every
// standard level supported", so a policy that cannot transmit any level has to deny each one
// explicitly. It returns a fresh map so callers may edit it.
func NoTransmissibleLevels() ai.ThinkingLevelMap {
	return levelsOf("", "", "", "", "", "", "")
}

// responsesEfforts are the efforts the Responses API accepts. `none` is the disable spelling and
// `max` is a real Responses effort and not a Chat-only value.
var responsesEfforts = map[string]bool{"none": true, "minimal": true, "low": true, "medium": true, "high": true, "xhigh": true, "max": true}

// SerializerPolicy closes a level map with the compat that serializes it, so a consumer cannot take
// a level map without the conclusion that carries it.
type SerializerPolicy struct {
	Reasoning        bool
	ThinkingLevelMap ai.ThinkingLevelMap
	Compat           *ai.ModelCompat
}

// chatCarrier reports explicit evidence only; the caller sets the carrier it concluded.
func chatCarrier(compat *ai.ModelCompat) bool {
	return compat != nil && (compat.ThinkingFormat != "" || isTrue(compat.SupportsReasoningEffort))
}

func deniesChatEffort(compat *ai.ModelCompat) bool {
	return compat != nil && isFalse(compat.SupportsReasoningEffort) && compat.ThinkingFormat == ""
}

// meetVendorCompat fields. Restrictions that only remove unsupported request fields are safe to
// apply as soon as one deployment requires them. Shape-changing fields need every deployment to
// agree because an unlabeled sibling may reject the alternate wire form.
var compatApplyIfAny = []struct {
	get func(*ai.ModelCompat) *bool
	set func(*ai.ModelCompat, *bool)
}{
	{func(c *ai.ModelCompat) *bool { return c.SupportsStore }, func(c *ai.ModelCompat, v *bool) { c.SupportsStore = v }},
	{func(c *ai.ModelCompat) *bool { return c.SupportsDeveloperRole }, func(c *ai.ModelCompat, v *bool) { c.SupportsDeveloperRole = v }},
	{func(c *ai.ModelCompat) *bool { return c.SupportsReasoningEffort }, func(c *ai.ModelCompat, v *bool) { c.SupportsReasoningEffort = v }},
	{func(c *ai.ModelCompat) *bool { return c.SupportsStrictMode }, func(c *ai.ModelCompat, v *bool) { c.SupportsStrictMode = v }},
}

// MeetVendorCompat combines per-deployment vendor compat; nil entries are deployments with none.
func MeetVendorCompat(perDeployment []*ai.ModelCompat) *ai.ModelCompat {
	if len(perDeployment) == 0 {
		return nil
	}
	var met ai.ModelCompat
	any := false
	field := func(compat *ai.ModelCompat, get func(*ai.ModelCompat) *bool) *bool {
		if compat == nil {
			return nil
		}
		return get(compat)
	}
	for _, f := range compatApplyIfAny {
		var stated []*bool
		for _, compat := range perDeployment {
			if v := field(compat, f.get); v != nil {
				stated = append(stated, v)
			}
		}
		if len(stated) > 0 && allSameBool(stated) {
			f.set(&met, stated[0])
			any = true
		}
	}
	stringFields := []struct {
		get func(*ai.ModelCompat) string
		set func(*ai.ModelCompat, string)
	}{
		{func(c *ai.ModelCompat) string { return c.MaxTokensField }, func(c *ai.ModelCompat, v string) { c.MaxTokensField = v }},
		{func(c *ai.ModelCompat) string { return c.CacheControlFormat }, func(c *ai.ModelCompat, v string) { c.CacheControlFormat = v }},
		{func(c *ai.ModelCompat) string { return c.ThinkingFormat }, func(c *ai.ModelCompat, v string) { c.ThinkingFormat = v }},
	}
	for _, f := range stringFields {
		first := ""
		agreed := true
		for i, compat := range perDeployment {
			value := ""
			if compat != nil {
				value = f.get(compat)
			}
			if value == "" || (i > 0 && value != first) {
				agreed = false
				break
			}
			first = value
		}
		if agreed {
			f.set(&met, first)
			any = true
		}
	}
	replay := make([]*bool, len(perDeployment))
	replayAll := true
	for i, compat := range perDeployment {
		replay[i] = field(compat, func(c *ai.ModelCompat) *bool { return c.RequiresReasoningContentOnAssistantMessages })
		replayAll = replayAll && replay[i] != nil
	}
	if replayAll && allSameBool(replay) {
		met.RequiresReasoningContentOnAssistantMessages = replay[0]
		any = true
	}
	if !any {
		return nil
	}
	return &met
}

func allSameBool(values []*bool) bool {
	for _, value := range values {
		if *value != *values[0] {
			return false
		}
	}
	return true
}

// ToResponsesLevels translates a Chat-shaped map into Responses efforts, denying any level with no
// valid Responses value instead of letting it through verbatim. pi-ai treats an absent xhigh/max
// entry as unsupported (unlike the standard levels), so the translation preserves that distinction.
func ToResponsesLevels(levels ai.ThinkingLevelMap) ai.ThinkingLevelMap {
	translated := ai.ThinkingLevelMap{}
	for _, level := range extendedLevels {
		chat, has := levels[level]
		if level == ai.ThinkingOff {
			// Disable is `none` on this API, never `off`.
			if has && chat == nil {
				translated[level] = nil
			} else {
				translated[level] = stringPtr("none")
			}
			continue
		}
		if !has && (level == ai.ThinkingXHigh || level == ai.ThinkingMax) {
			translated[level] = nil
			continue
		}
		// pi-ai passes an absent standard level through as its own name.
		candidate := chat
		if !has {
			candidate = stringPtr(string(level))
		}
		if candidate != nil && responsesEfforts[*candidate] {
			translated[level] = candidate
		} else {
			translated[level] = nil
		}
	}
	return translated
}

// CloseSerializerPolicyInput is the input of CloseSerializerPolicy.
type CloseSerializerPolicyInput struct {
	API            types.LiteLLMApi
	Reasoning      bool
	VendorCompat   *ai.ModelCompat
	SemanticCompat *ReasoningCompat
	SemanticLevels ai.ThinkingLevelMap
	CatalogLevels  ai.ThinkingLevelMap
	// RequireChatCarrier denies levels when no candidate map exists.
	RequireChatCarrier bool
	// AllowInferredChatCarrier nil means true.
	AllowInferredChatCarrier         *bool
	AcceptsResponsesReasoningControl bool
	// DenyLevels is for callers with no level evidence at all. It states the no-level conclusion
	// explicitly, so a caller spreading the policy over an earlier model cannot leave a stale map.
	DenyLevels bool
}

// CloseSerializerPolicy is the one place a level map is paired with a serializer conclusion. Every
// discovery, cache, health and singleton path consumes the whole returned object.
func CloseSerializerPolicy(input CloseSerializerPolicyInput) SerializerPolicy {
	reasoning := input.Reasoning
	vendor := input.VendorCompat
	allowInferred := input.AllowInferredChatCarrier == nil || *input.AllowInferredChatCarrier

	switch input.API {
	case types.APIAnthropicMessages:
		var compat *ai.ModelCompat
		if vendor != nil {
			picked := ai.ModelCompat{
				ForceAdaptiveThinking: vendor.ForceAdaptiveThinking,
				SupportsTemperature:   vendor.SupportsTemperature,
				SupportsStrictTools:   vendor.SupportsStrictTools,
			}
			if !reflect.DeepEqual(picked, ai.ModelCompat{}) {
				compat = &picked
			}
		}
		policy := SerializerPolicy{Reasoning: reasoning, Compat: compat}
		if reasoning && (input.DenyLevels || input.CatalogLevels != nil) {
			if input.DenyLevels {
				policy.ThinkingLevelMap = NoTransmissibleLevels()
			} else {
				policy.ThinkingLevelMap = input.CatalogLevels
			}
		}
		return policy
	case types.APIOpenAIResponses:
		// Responses always serializes a selected level as `reasoning.effort`. Chat-only evidence such
		// as `thinking` cannot authorize that carrier.
		var compat *ai.ModelCompat
		if vendor != nil {
			picked := ai.ModelCompat{
				SupportsDeveloperRole:           vendor.SupportsDeveloperRole,
				SessionAffinityFormat:           vendor.SessionAffinityFormat,
				SupportsLongCacheRetention:      vendor.SupportsLongCacheRetention,
				SupportsStrictMode:              vendor.SupportsStrictMode,
				SupportsOpenAIGrammarTools:      vendor.SupportsOpenAIGrammarTools,
				SupportsAdditionalTools:         vendor.SupportsAdditionalTools,
				SupportsToolSearch:              vendor.SupportsToolSearch,
				SupportsExplicitPromptCacheMode: vendor.SupportsExplicitPromptCacheMode,
			}
			if !reflect.DeepEqual(picked, ai.ModelCompat{}) {
				compat = &picked
			}
		}
		if !reasoning {
			return SerializerPolicy{Reasoning: reasoning, Compat: compat}
		}
		if input.DenyLevels || !input.AcceptsResponsesReasoningControl {
			return SerializerPolicy{Reasoning: reasoning, ThinkingLevelMap: NoTransmissibleLevels(), Compat: compat}
		}
		levels := input.SemanticLevels
		if levels == nil {
			levels = input.CatalogLevels
		}
		return SerializerPolicy{Reasoning: reasoning, ThinkingLevelMap: ToResponsesLevels(levels), Compat: compat}
	}

	// A model with no compat at all keeps none: fabricating an empty object here would rewrite every
	// cached model that passes through.
	stated := vendor != nil || input.SemanticCompat != nil
	var merged ai.ModelCompat
	if vendor != nil {
		merged = *vendor
	}
	if semantic := input.SemanticCompat; semantic != nil {
		if semantic.RequiresReasoningContentOnAssistantMessages != nil {
			merged.RequiresReasoningContentOnAssistantMessages = semantic.RequiresReasoningContentOnAssistantMessages
		}
		if semantic.SupportsReasoningEffort != nil {
			merged.SupportsReasoningEffort = semantic.SupportsReasoningEffort
		}
		if semantic.ThinkingFormat != "" {
			merged.ThinkingFormat = semantic.ThinkingFormat
		}
	}
	var compat *ai.ModelCompat
	if stated {
		copied := merged
		compat = &copied
	}
	denied := merged
	denied.SupportsReasoningEffort = boolPtr(false)
	if !reasoning {
		if input.DenyLevels {
			return SerializerPolicy{Reasoning: reasoning, Compat: &denied}
		}
		return SerializerPolicy{Reasoning: reasoning, Compat: compat}
	}
	if input.DenyLevels {
		return SerializerPolicy{Reasoning: reasoning, ThinkingLevelMap: NoTransmissibleLevels(), Compat: &denied}
	}
	if deniesChatEffort(&merged) {
		// The vendor denies effort and named no format: nothing can carry a level.
		return SerializerPolicy{Reasoning: reasoning, ThinkingLevelMap: NoTransmissibleLevels(), Compat: compat}
	}
	candidate := input.SemanticLevels
	if candidate == nil {
		candidate = input.CatalogLevels
	}
	if candidate == nil {
		// An absent map enables pi-ai's standard levels, so freshly discovered routes without an
		// explicit carrier must be represented as a denial.
		if input.RequireChatCarrier {
			return SerializerPolicy{Reasoning: reasoning, ThinkingLevelMap: NoTransmissibleLevels(), Compat: compat}
		}
		return SerializerPolicy{Reasoning: reasoning, Compat: compat}
	}
	if chatCarrier(&merged) {
		return SerializerPolicy{Reasoning: reasoning, ThinkingLevelMap: candidate, Compat: compat}
	}
	if !allowInferred {
		return SerializerPolicy{Reasoning: reasoning, ThinkingLevelMap: NoTransmissibleLevels(), Compat: &denied}
	}
	// Callers that explicitly permit inference may treat their own direct router evidence as a
	// carrier. Route, catalog, and legacy cache evidence must opt out.
	inferred := merged
	inferred.SupportsReasoningEffort = boolPtr(true)
	return SerializerPolicy{Reasoning: reasoning, ThinkingLevelMap: candidate, Compat: &inferred}
}

func failClosedReasoning(reasoning bool, replay *bool) ReasoningPolicy {
	policy := ReasoningPolicy{
		Reasoning: reasoning,
		Compat:    &ReasoningCompat{RequiresReasoningContentOnAssistantMessages: replay, SupportsReasoningEffort: boolPtr(false)},
	}
	if reasoning {
		policy.ThinkingLevelMap = NoTransmissibleLevels()
	}
	return policy
}

func buildReasoningPolicy(
	semanticModel SemanticModel,
	acceptedOpenAIParams []string,
	reducedReasoning bool,
	explicitlyUnsupported bool,
) ReasoningPolicy {
	acceptsThinking := slices.Contains(acceptedOpenAIParams, thinkingParam)
	acceptsEffort := slices.Contains(acceptedOpenAIParams, reasoningEffortParam)
	hasAcceptedControl := acceptsThinking || acceptsEffort
	reasoning := !explicitlyUnsupported && (reducedReasoning || hasAcceptedControl || semanticModel == SemanticModelKimiK27Code)
	withOff := func(off string) ai.ThinkingLevelMap {
		levels := onlyHigh()
		levels[ai.ThinkingOff] = stringPtr(off)
		return levels
	}
	replay := func() *bool { return boolPtr(true) }

	switch semanticModel {
	case SemanticModelKimiK25K26:
		// Binary thinking rides the `thinking` param, so without that accepted param there is no wire
		// mechanism and every level must be denied.
		if acceptsThinking && reasoning {
			return ReasoningPolicy{
				Reasoning:        reasoning,
				ThinkingLevelMap: withOff("off"),
				Compat:           &ReasoningCompat{ThinkingFormat: "deepseek", SupportsReasoningEffort: boolPtr(false)},
			}
		}
		return failClosedReasoning(reasoning, nil)
	case SemanticModelKimiK27Code:
		// K2.7 Code always reasons and cannot be switched off, so `off` stays denied rather than
		// inventing a disable control. The `high` level only exists when a deployment accepts
		// `thinking` to carry it.
		if acceptsThinking && reasoning {
			return ReasoningPolicy{
				Reasoning:        reasoning,
				ThinkingLevelMap: onlyHigh(),
				Compat: &ReasoningCompat{
					RequiresReasoningContentOnAssistantMessages: replay(),
					ThinkingFormat:          "deepseek",
					SupportsReasoningEffort: boolPtr(false),
				},
			}
		}
		return failClosedReasoning(reasoning, replay())
	case SemanticModelKimiK3:
		if acceptsEffort && reasoning {
			return ReasoningPolicy{
				Reasoning:        reasoning,
				ThinkingLevelMap: levelsOf("", "", "low", "", "high", "", "max"),
				Compat: &ReasoningCompat{
					RequiresReasoningContentOnAssistantMessages: replay(),
					ThinkingFormat:          "openai",
					SupportsReasoningEffort: boolPtr(true),
				},
			}
		}
		return failClosedReasoning(reasoning, replay())
	case SemanticModelDeepSeekV4:
		if acceptsThinking && acceptsEffort && reasoning {
			return ReasoningPolicy{
				Reasoning:        reasoning,
				ThinkingLevelMap: levelsOf("off", "", "", "", "high", "", "max"),
				Compat: &ReasoningCompat{
					RequiresReasoningContentOnAssistantMessages: replay(),
					ThinkingFormat:          "deepseek",
					SupportsReasoningEffort: boolPtr(true),
				},
			}
		}
		if acceptsEffort && reasoning {
			return ReasoningPolicy{
				Reasoning:        reasoning,
				ThinkingLevelMap: levelsOf("", "", "", "", "high", "", "max"),
				Compat: &ReasoningCompat{
					RequiresReasoningContentOnAssistantMessages: replay(),
					ThinkingFormat:          "openai",
					SupportsReasoningEffort: boolPtr(true),
				},
			}
		}
		if acceptsThinking && reasoning {
			return ReasoningPolicy{
				Reasoning:        reasoning,
				ThinkingLevelMap: withOff("off"),
				Compat: &ReasoningCompat{
					RequiresReasoningContentOnAssistantMessages: replay(),
					ThinkingFormat:          "deepseek",
					SupportsReasoningEffort: boolPtr(false),
				},
			}
		}
		return failClosedReasoning(reasoning, replay())
	}
	// No identified semantic model means no known reasoning contract; the caller ignores this policy
	// entirely and keeps the reduced/catalog metadata.
	return ReasoningPolicy{Reasoning: false}
}

// explicitCost reads a router price per million tokens; negative and non-finite prices are unresolved.
func explicitCost(entry types.ModelInfoEntry, field costField) *float64 {
	info := entry.ModelInfo
	if info == nil {
		return nil
	}
	var perToken *float64
	switch field {
	case costInput:
		perToken = info.InputCostPerToken
	case costOutput:
		perToken = info.OutputCostPerToken
	case costCacheRead:
		perToken = info.CacheReadInputTokenCost
	case costCacheWrite:
		perToken = info.CacheCreationInputTokenCost
	}
	if perToken == nil || !finite(*perToken) || *perToken < 0 {
		return nil
	}
	value := *perToken * 1_000_000
	return &value
}

func catalogCostField(catalog *CatalogResolution, field costField) *float64 {
	if catalog == nil || catalog.Cost == nil {
		return nil
	}
	switch field {
	case costInput:
		return catalog.Cost.Input
	case costOutput:
		return catalog.Cost.Output
	case costCacheRead:
		return catalog.Cost.CacheRead
	}
	return catalog.Cost.CacheWrite
}

func resolvedCost(entry types.ModelInfoEntry, catalog *CatalogResolution, field costField) *float64 {
	if explicit := explicitCost(entry, field); explicit != nil {
		return explicit
	}
	return catalogCostField(catalog, field)
}

func minOf(values []float64) float64 {
	out := math.Inf(1)
	for _, value := range values {
		out = math.Min(out, value)
	}
	return out
}

func maxOf(values ...float64) float64 {
	out := math.Inf(-1)
	for _, value := range values {
		out = math.Max(out, value)
	}
	return out
}

func conservativeLimit(explicit, catalog *float64) *float64 {
	var valid []float64
	for _, candidate := range []*float64{explicitLimit(explicit), explicitLimit(catalog)} {
		if candidate != nil {
			valid = append(valid, *candidate)
		}
	}
	if len(valid) == 0 {
		return nil
	}
	out := minOf(valid)
	return &out
}

// unanimous returns the first value when every value equals it; the empty string is absent.
func unanimous(values []string) string {
	if len(values) == 0 || values[0] == "" {
		return ""
	}
	for _, value := range values {
		if value != values[0] {
			return ""
		}
	}
	return values[0]
}

func ratesAboveThreshold(cost ai.ModelCost, threshold int) ai.ModelCost {
	matchedThreshold := -1
	var matches []ai.CostTier
	for _, tier := range cost.Tiers {
		if tier.InputTokensAbove < 0 || tier.InputTokensAbove > threshold {
			continue
		}
		if tier.InputTokensAbove > matchedThreshold {
			matchedThreshold = tier.InputTokensAbove
			matches = []ai.CostTier{tier}
		} else if tier.InputTokensAbove == matchedThreshold {
			// Pi uses the first duplicate threshold. Taking the per-field maximum is conservative if
			// malformed catalog data supplies conflicting duplicates.
			matches = append(matches, tier)
		}
	}
	if len(matches) == 0 {
		return cost
	}
	out := ai.ModelCost{Input: cost.Input, Output: cost.Output, CacheRead: cost.CacheRead, CacheWrite: cost.CacheWrite}
	for _, tier := range matches {
		out.Input = math.Max(out.Input, tier.InputCostPer1M)
		out.Output = math.Max(out.Output, tier.OutputCostPer1M)
		out.CacheRead = math.Max(out.CacheRead, tier.CacheReadCostPer1M)
		out.CacheWrite = math.Max(out.CacheWrite, tier.CacheWriteCostPer1M)
	}
	return out
}

// ConservativeCostTiers builds the per-field upper envelope of complete request-wide price ladders.
// Every source threshold is retained because crossing it can change which deployment is most
// expensive, even when the ladders use different breakpoints.
func ConservativeCostTiers(costs []ai.ModelCost) []ai.CostTier {
	seen := map[int]bool{}
	var thresholds []int
	for _, cost := range costs {
		for _, tier := range cost.Tiers {
			if tier.InputTokensAbove >= 0 && !seen[tier.InputTokensAbove] {
				seen[tier.InputTokensAbove] = true
				thresholds = append(thresholds, tier.InputTokensAbove)
			}
		}
	}
	if len(thresholds) == 0 {
		return nil
	}
	sort.Ints(thresholds)
	out := make([]ai.CostTier, len(thresholds))
	for i, threshold := range thresholds {
		tier := ai.CostTier{InputTokensAbove: threshold}
		for j, cost := range costs {
			rate := ratesAboveThreshold(cost, threshold)
			if j == 0 {
				tier.InputCostPer1M, tier.OutputCostPer1M = rate.Input, rate.Output
				tier.CacheReadCostPer1M, tier.CacheWriteCostPer1M = rate.CacheRead, rate.CacheWrite
				continue
			}
			tier.InputCostPer1M = math.Max(tier.InputCostPer1M, rate.Input)
			tier.OutputCostPer1M = math.Max(tier.OutputCostPer1M, rate.Output)
			tier.CacheReadCostPer1M = math.Max(tier.CacheReadCostPer1M, rate.CacheRead)
			tier.CacheWriteCostPer1M = math.Max(tier.CacheWriteCostPer1M, rate.CacheWrite)
		}
		out[i] = tier
	}
	return out
}

type canonicalDeployments struct {
	candidates     []types.ModelInfoEntry
	candidateModes []Mode
	routable       []types.ModelInfoEntry
}

func entryMode(entry types.ModelInfoEntry) Mode {
	if entry.ModelInfo == nil {
		return ModeUnknown
	}
	return NormalizedMode(entry.ModelInfo.Mode)
}

func canonicalModelInfoDeployments(entries []types.ModelInfoEntry) canonicalDeployments {
	// A group is addressed by its public route name, so a row without a readable one cannot
	// participate. Enforce that here so every authority consumer observes the same filtered and
	// deduplicated set.
	var named []types.ModelInfoEntry
	for _, entry := range entries {
		if entry.ModelName != "" {
			named = append(named, entry)
		}
	}
	out := canonicalDeployments{candidates: uniqueDeployments(named)}
	for _, entry := range out.candidates {
		mode := entryMode(entry)
		out.candidateModes = append(out.candidateModes, mode)
		if mode != ModeUnsupported {
			out.routable = append(out.routable, entry)
		}
	}
	return out
}

// HasMixedIncompatibleDeploymentModes reports a group mixing chat-style and unsupported deployments.
func HasMixedIncompatibleDeploymentModes(entries []types.ModelInfoEntry) bool {
	modes := canonicalModelInfoDeployments(entries).candidateModes
	hasUnsupported, hasOther := false, false
	for _, mode := range modes {
		if mode == ModeUnsupported {
			hasUnsupported = true
		} else {
			hasOther = true
		}
	}
	return hasUnsupported && hasOther
}

var (
	kimiFamilyPattern     = regexp.MustCompile(`(?i)(?:^|[./_-])(?:moonshotai|moonshot|kimi)(?:$|[./_:-])`)
	forcedThinkingPattern = regexp.MustCompile(`(?i)(?:^|[./_-])thinking(?:$|[./_:-])`)
	moonshotRoutes        = map[string]bool{"moonshot": true, "moonshotai": true}
)

type kimiEvidence struct {
	identified, forcedThinking, moonshotTransport bool
}

func backendRow(entry types.ModelInfoEntry) backend.Row {
	row := backend.Row{ModelName: entry.ModelName}
	if entry.LiteLLMParams != nil {
		row.LiteLLMParams = backend.RowParams{Model: entry.LiteLLMParams.Model, CustomLLMProvider: entry.LiteLLMParams.CustomLLMProvider}
	}
	if entry.ModelInfo != nil {
		row.ModelInfo = backend.RowInfo{BaseModel: entry.ModelInfo.BaseModel, LiteLLMProvider: entry.ModelInfo.LiteLLMProvider}
	}
	return row
}

func kimiDeploymentEvidence(entry types.ModelInfoEntry) kimiEvidence {
	row := backendRow(entry)
	evidence := kimiEvidence{}
	for _, candidate := range []string{row.LiteLLMParams.Model, row.LiteLLMParams.CustomLLMProvider, row.ModelInfo.BaseModel, row.ModelInfo.LiteLLMProvider} {
		identity := strings.TrimSpace(candidate)
		if identity != "" && kimiFamilyPattern.MatchString(identity) {
			evidence.identified = true
			evidence.forcedThinking = evidence.forcedThinking || forcedThinkingPattern.MatchString(identity)
		}
	}
	// Visibility parameters are accepted by Moonshot's API, not by every host that serves a Kimi
	// model. Both declared routing signals must name Moonshot.
	evidence.moonshotTransport = backend.RoutesOnlyThrough(row, moonshotRoutes)
	return evidence
}

func infoOf(entry types.ModelInfoEntry) *types.ModelInfoDetails {
	if entry.ModelInfo == nil {
		return &types.ModelInfoDetails{}
	}
	return entry.ModelInfo
}

// ReduceModelGroup reduces every deployment behind one public route into a single model, or nil when
// the group is empty or contains a non-chat deployment.
func ReduceModelGroup(entries []types.ModelInfoEntry, resolveCatalog CatalogResolver) *ReducedModelGroup {
	canonical := canonicalModelInfoDeployments(entries)
	candidates, candidateModes, deployments := canonical.candidates, canonical.candidateModes, canonical.routable
	if len(candidates) == 0 {
		return nil
	}
	// Every deployment behind a public route must accept a chat-style request. Dropping an
	// explicitly incompatible sibling would publish a route that can still select an embedding or
	// other non-chat deployment.
	if slices.Contains(candidateModes, ModeUnsupported) {
		return nil
	}
	catalogs := make([]*CatalogResolution, len(deployments))
	for i, entry := range deployments {
		catalogs[i] = resolveCatalog(entry)
	}
	pick := func(get func(*CatalogResolution) string) []string {
		out := make([]string, len(catalogs))
		for i, catalog := range catalogs {
			if catalog != nil {
				out[i] = get(catalog)
			}
		}
		return out
	}
	catalogProvider := unanimous(pick(func(c *CatalogResolution) string { return c.Provider }))
	catalogModelIDs := pick(func(c *CatalogResolution) string { return c.CatalogModelID })
	hasCatalogModelIdentity := slices.ContainsFunc(catalogModelIDs, func(id string) bool { return id != "" })
	catalogModelID := unanimous(catalogModelIDs)
	// Provider-only test resolvers preserve the reducer's public unit-test contract; production
	// catalog resolutions always include the concrete model identity.
	hasCatalogAuthority := catalogProvider != "" && (!hasCatalogModelIdentity || catalogModelID != "")
	catalogAuthority := make([]*CatalogResolution, len(catalogs))
	catalogAuthorityAmbiguous := false
	if hasCatalogAuthority {
		copy(catalogAuthority, catalogs)
	} else {
		for _, catalog := range catalogs {
			if catalog != nil && (catalog.Provider != "" || catalog.CatalogModelID != "") {
				catalogAuthorityAmbiguous = true
			}
		}
	}
	semanticModel := SemanticModel(unanimous(pick(func(c *CatalogResolution) string { return string(c.SemanticModel) })))
	semanticFamily := FamilyEvidence(unanimous(pick(func(c *CatalogResolution) string { return string(c.SemanticFamily) })))
	messagesCompatJSON := unanimous(pick(func(c *CatalogResolution) string {
		encoded, _ := StableJSON(c.MessagesCompat)
		return encoded
	}))
	// Strict tools require affirmative evidence from every routable deployment.
	messagesStrictTools := len(catalogs) > 0
	for _, catalog := range catalogs {
		messagesStrictTools = messagesStrictTools && catalog != nil && catalog.MessagesStrictTools
	}
	// LiteLLM v1.100+ reports `supported_endpoints: null` for every model its model map does not
	// list, so null means unknown, like an absent field. Only a list without Messages withholds the
	// transport.
	messagesEndpointAllowed := true
	for _, entry := range deployments {
		endpoints := infoOf(entry).SupportedEndpoints
		messagesEndpointAllowed = messagesEndpointAllowed && (endpoints == nil || slices.Contains(endpoints, messagesEndpoint))
	}
	allModes := func(want Mode) bool {
		for _, mode := range candidateModes {
			if mode != want {
				return false
			}
		}
		return true
	}
	api := types.APIOpenAICompletions
	switch {
	case allModes(ModeResponses):
		api = types.APIOpenAIResponses
	case allModes(ModeChat) && messagesEndpointAllowed && semanticFamily == SemanticFamilyClaude && messagesCompatJSON != "":
		api = types.APIAnthropicMessages
	}
	n := len(deployments)
	reasoningEvidence := make([]*bool, n)
	visionEvidence := make([]*bool, n)
	contextWindowEvidence := make([]*float64, n)
	maxTokensEvidence := make([]*float64, n)
	for i, entry := range deployments {
		info := infoOf(entry)
		authority := catalogAuthority[i]
		reasoningEvidence[i], visionEvidence[i] = info.SupportsReasoning, info.SupportsVision
		var catalogContext, catalogMax *float64
		if authority != nil {
			if reasoningEvidence[i] == nil {
				reasoningEvidence[i] = authority.Reasoning
			}
			if visionEvidence[i] == nil {
				visionEvidence[i] = authority.Vision
			}
			catalogContext, catalogMax = authority.ContextWindow, authority.MaxTokens
		}
		contextWindowEvidence[i] = conservativeLimit(info.MaxInputTokens, catalogContext)
		maxTokensEvidence[i] = conservativeLimit(info.MaxOutputTokens, catalogMax)
	}
	everyTrue := func(values []*bool) bool {
		for _, value := range values {
			if !isTrue(value) {
				return false
			}
		}
		return true
	}
	allDefined := func(values []*bool) bool {
		return !slices.Contains(values, nil)
	}
	allDefinedFloat := func(values []*float64) bool {
		return !slices.Contains(values, nil)
	}
	reasoning := everyTrue(reasoningEvidence)
	explicitlyUnsupported := false
	for _, entry := range deployments {
		explicitlyUnsupported = explicitlyUnsupported || isFalse(infoOf(entry).SupportsReasoning)
	}
	vision := everyTrue(visionEvidence)
	fallbackContextWindow := float64(DefaultContextWindow())
	limitMin := func(values []*float64, fallback float64) float64 {
		resolved := make([]float64, len(values))
		for i, value := range values {
			resolved[i] = fallback
			if value != nil {
				resolved[i] = *value
			}
		}
		return minOf(resolved)
	}
	contextWindow := limitMin(contextWindowEvidence, fallbackContextWindow)
	maxTokens := limitMin(maxTokensEvidence, DefaultMaxTokens)

	var costValues [costFieldCount][]*float64
	var completeCostFields [costFieldCount]bool
	hasCompleteCost := true
	var costTotals [costFieldCount]float64
	for field := costInput; field < costFieldCount; field++ {
		costValues[field] = make([]*float64, n)
		complete := true
		resolved := make([]float64, 0, n)
		for i, entry := range deployments {
			costValues[field][i] = resolvedCost(entry, catalogAuthority[i], field)
			if costValues[field][i] == nil {
				complete = false
			} else {
				resolved = append(resolved, *costValues[field][i])
			}
		}
		completeCostFields[field] = complete
		hasCompleteCost = hasCompleteCost && complete
		if complete {
			costTotals[field] = maxOf(resolved...)
		}
	}
	// A defaulted capability or limit is as much a gap as a missing price: the incomplete marker
	// covers everything the reducer had to assume.
	hasCompleteMetadata := hasCompleteCost && allDefined(reasoningEvidence) && allDefined(visionEvidence) &&
		allDefinedFloat(contextWindowEvidence) && allDefinedFloat(maxTokensEvidence)
	cost := ai.ModelCost{
		Input:      costTotals[costInput],
		Output:     costTotals[costOutput],
		CacheRead:  costTotals[costCacheRead],
		CacheWrite: costTotals[costCacheWrite],
	}
	if hasCompleteCost && hasCatalogAuthority {
		deploymentCosts := make([]ai.ModelCost, n)
		for i, entry := range deployments {
			base := ai.ModelCost{
				Input:      *costValues[costInput][i],
				Output:     *costValues[costOutput][i],
				CacheRead:  *costValues[costCacheRead][i],
				CacheWrite: *costValues[costCacheWrite][i],
			}
			var explicit [costFieldCount]bool
			explicitCount := 0
			for field := costInput; field < costFieldCount; field++ {
				if explicitCost(entry, field) != nil {
					explicit[field] = true
					explicitCount++
				}
			}
			var catalogTiers []ai.CostTier
			if authority := catalogAuthority[i]; authority != nil && authority.Cost != nil {
				catalogTiers = authority.Cost.Tiers
			}
			// An explicit field replaces catalog pricing for that field at every threshold. Unaffected
			// fields retain their catalog ladder; otherwise a partial router override could hide a
			// known higher catalog rate.
			if catalogTiers != nil && explicitCount < int(costFieldCount) {
				base.Tiers = make([]ai.CostTier, len(catalogTiers))
				for j, tier := range catalogTiers {
					replaced := tier
					if explicit[costInput] {
						replaced.InputCostPer1M = base.Input
					}
					if explicit[costOutput] {
						replaced.OutputCostPer1M = base.Output
					}
					if explicit[costCacheRead] {
						replaced.CacheReadCostPer1M = base.CacheRead
					}
					if explicit[costCacheWrite] {
						replaced.CacheWriteCostPer1M = base.CacheWrite
					}
					base.Tiers[j] = replaced
				}
			}
			deploymentCosts[i] = base
		}
		if tiers := ConservativeCostTiers(deploymentCosts); tiers != nil {
			cost.Tiers = tiers
		}
	}
	acceptedOpenAIParams := intersectParams(deployments)
	// Kimi and DeepSeek generations name their own carriers, so an operator opt-in cannot substitute
	// for the parameter evidence their contracts require.
	namedCarrierFamily := false
	for _, catalog := range catalogs {
		if catalog != nil {
			switch catalog.SemanticFamily {
			case SemanticFamilyKimi, SemanticFamilyDeepSeek, FamilyConflicting:
				namedCarrierFamily = true
			}
		}
	}
	acceptsResponsesReasoningControl := slices.Contains(acceptedOpenAIParams, reasoningEffortParam)
	if !acceptsResponsesReasoningControl && !namedCarrierFamily && n > 0 {
		acceptsResponsesReasoningControl = true
		for _, entry := range deployments {
			if !acceptedParams(entry)[reasoningEffortParam] && !optsIntoEffortCarrier(entry) {
				acceptsResponsesReasoningControl = false
			}
		}
	}
	publicEfforts := make([][]string, n)
	authorityMaps := make([]ai.ThinkingLevelMap, n)
	for i, authority := range catalogAuthority {
		if authority != nil {
			publicEfforts[i], authorityMaps[i] = authority.EffortLevels, authority.ThinkingLevelMap
		}
	}
	evidenceLevelMap := reasoningLevelMap(deployments, publicEfforts, []ai.ThinkingLevelMap{thinking.Intersect(authorityMaps)})
	var thinkingLevelMap ai.ThinkingLevelMap
	if reasoning && acceptsResponsesReasoningControl {
		thinkingLevelMap = evidenceLevelMap
	}
	if api == types.APIAnthropicMessages {
		messagesMaps := make([]ai.ThinkingLevelMap, len(catalogs))
		for i, catalog := range catalogs {
			if catalog != nil {
				messagesMaps[i] = catalog.MessagesThinkingLevelMap
			}
		}
		thinkingLevelMap = thinking.Intersect(messagesMaps)
		// Native serializer restrictions apply even when catalog pricing is withheld. Router flags
		// can also deny Pi's implicit default levels, never add a level.
		for _, level := range extendedLevels {
			anyFalse, anyDefined, allTrue := false, false, true
			for _, entry := range deployments {
				reported := reportedLevel(entry, level)
				anyFalse = anyFalse || isFalse(reported)
				anyDefined = anyDefined || reported != nil
				allTrue = allTrue && isTrue(reported)
			}
			if anyFalse || ((level == ai.ThinkingXHigh || level == ai.ThinkingMax) && anyDefined && !allTrue) {
				if thinkingLevelMap == nil {
					thinkingLevelMap = ai.ThinkingLevelMap{}
				}
				thinkingLevelMap[level] = nil
			}
		}
	}
	unanimousNormalKimi, unanimousMoonshotTransport := true, true
	for _, entry := range deployments {
		evidence := kimiDeploymentEvidence(entry)
		unanimousNormalKimi = unanimousNormalKimi && evidence.identified && !evidence.forcedThinking
		unanimousMoonshotTransport = unanimousMoonshotTransport && evidence.moonshotTransport
	}

	if n == 0 || deployments[0].ModelName == "" {
		return nil
	}
	id := deployments[0].ModelName

	semanticReasoningPolicy := buildReasoningPolicy(semanticModel, acceptedOpenAIParams, reasoning, explicitlyUnsupported)
	// A semantic policy that denies the effort carrier describes a generation with binary or fixed
	// thinking. When no public catalog states effort levels, that contract stays in force even
	// though a host accepts `reasoning_effort`; otherwise Pi's defaults would turn a single on/off
	// switch into five selectable levels. Positive public level evidence still governs, as it does
	// for level-based generations whose accepted levels vary by host: Moonshot rejects `medium` on
	// K3 while Azure Foundry accepts it.
	semanticMap := semanticReasoningPolicy.ThinkingLevelMap
	hasEvidenceLevels := false
	for _, level := range evidenceLevelMap {
		hasEvidenceLevels = hasEvidenceLevels || level != nil
	}
	semanticBinary := semanticReasoningPolicy.Compat != nil &&
		isFalse(semanticReasoningPolicy.Compat.SupportsReasoningEffort) && !hasEvidenceLevels
	explicitOffDenial := false
	for _, entry := range deployments {
		explicitOffDenial = explicitOffDenial || isFalse(reportedLevel(entry, ai.ThinkingOff))
	}
	reasoningPolicy := semanticReasoningPolicy
	if semanticMap != nil {
		switch {
		case !acceptsResponsesReasoningControl:
			reasoningPolicy.ThinkingLevelMap = semanticMap
		case semanticBinary:
			merged := ai.ThinkingLevelMap{}
			for level, value := range semanticMap {
				merged[level] = value
			}
			for level := range deniedLevels(evidenceLevelMap) {
				merged[level] = nil
			}
			reasoningPolicy.ThinkingLevelMap = merged
		default:
			merged := ai.ThinkingLevelMap{}
			for level, value := range evidenceLevelMap {
				merged[level] = value
			}
			if off, has := semanticMap[ai.ThinkingOff]; has {
				if explicitOffDenial {
					merged[ai.ThinkingOff] = nil
				} else {
					merged[ai.ThinkingOff] = off
				}
			}
			reasoningPolicy.ThinkingLevelMap = merged
		}
	}

	group := &ReducedModelGroup{
		ID:                               id,
		API:                              api,
		Reasoning:                        reasoning,
		AcceptsResponsesReasoningControl: acceptsResponsesReasoningControl,
		ThinkingLevelMap:                 thinkingLevelMap,
		Vision:                           vision,
		ContextWindow:                    contextWindow,
		ContextWindowDefaulted:           slices.Contains(contextWindowEvidence, nil) && contextWindow == fallbackContextWindow,
		MaxTokens:                        maxTokens,
		Cost:                             cost,
		HasCompleteCost:                  hasCompleteCost,
		HasCompleteMetadata:              hasCompleteMetadata,
		SemanticModel:                    semanticModel,
		SemanticFamily:                   semanticFamily,
		MessagesStrictTools:              messagesStrictTools,
		CatalogAuthorityAmbiguous:        catalogAuthorityAmbiguous,
		DeploymentFamilies:               make([]FamilyEvidence, len(catalogs)),
		NormalizeThinkTags:               unanimousNormalKimi,
		SuppressReasoningVisibility:      unanimousNormalKimi && unanimousMoonshotTransport,
		AcceptedOpenAIParams:             acceptedOpenAIParams,
		ReasoningPolicy:                  reasoningPolicy,
	}
	if hasCatalogAuthority {
		group.CatalogProvider = catalogProvider
	}
	if messagesCompatJSON != "" {
		// Every catalog stated the same compat, so the first carries it.
		copied := *catalogs[0].MessagesCompat
		group.MessagesCompat = &copied
	}
	for i, catalog := range catalogs {
		if catalog != nil {
			group.DeploymentFamilies[i] = catalog.SemanticFamily
		}
	}
	return group
}

// CatalogResolutionFromModel is catalogResolution: it states a catalog model's evidence.
func CatalogResolutionFromModel(provider string, model *ai.Model) CatalogResolution {
	reasoning := model.ProviderMeta.Reasoning
	vision := slices.Contains(model.Input, "image")
	contextWindow := float64(model.Capabilities.ContextWindow)
	maxTokens := float64(model.Capabilities.MaxOutputTokens)
	capabilities := model.Capabilities
	return CatalogResolution{
		Provider:         provider,
		CatalogModelID:   model.ID,
		Reasoning:        &reasoning,
		ThinkingLevelMap: model.ThinkingLevelMap,
		Vision:           &vision,
		ContextWindow:    &contextWindow,
		MaxTokens:        &maxTokens,
		Cost: &CatalogCost{
			Input:      &capabilities.InputCostPer1M,
			Output:     &capabilities.OutputCostPer1M,
			CacheRead:  &capabilities.CacheReadCostPer1M,
			CacheWrite: &capabilities.CacheWriteCostPer1M,
			Tiers:      capabilities.CostTiers,
		},
	}
}
