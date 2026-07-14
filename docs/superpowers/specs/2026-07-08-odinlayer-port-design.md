# OdinLayer Behavior Port Design

**Status:** Approved

## Goal

Port useful behavior from `@odinlayer/pi-provider-litellm@1.0.19` into this extension without copying branding, release metadata, or UI/dependency churn.

## Scope

Implement three behaviors:

1. Discover LiteLLM Responses API models from `/model/info`.
2. Make LiteLLM MCP tool execution more resilient and parallel-friendly.
3. Support LiteLLM Skill Hub endpoints while keeping the existing Skills Gateway path working.

Out of scope:

- Compact MCP TUI rendering, because OdinLayer imports `@earendil-works/pi-tui` and this package currently has no runtime dependencies.
- Renaming `/litellm-refresh` to `/litellm:models-refresh`, because the existing command already supports all configured providers and changing it would be mostly user-facing churn.

## Design

### Responses API Discovery

`src/discover.ts` will treat `model_info.mode` values of `chat`, `response`, `responses`, or missing as chat-style models. It will continue filtering non-chat workloads such as embeddings. Response-mode models will get a per-model `api: "openai-responses"` override so Pi routes them through the Responses API while the provider default stays `openai-completions`.

Tests will cover direct `/model/info` entries and `/health` per-endpoint `/model/info` fallback entries.

### MCP Tools

`src/mcp-tools.ts` will mark generated MCP tools with `executionMode: "parallel"` and retry `executeMcpTool()` once for transient failures. Retryable failures are HTTP `408`, `425`, `429`, `500`, `502`, `503`, `504`, plus timeout/network errors. Auth and other non-retryable failures return immediately.

No new dependency is needed.

### Skill Hub Compatibility

`src/skills.ts` will first read Skill Hub catalog data from `GET /claude-code/marketplace.json`, accepting array bodies or `plugins`, `data`, or `skills` arrays. If that endpoint is unavailable, it will fall back to the existing `GET /v1/skills`.

Create/delete tools will use Skill Hub plugin endpoints by default:

- `POST /claude-code/plugins`
- `DELETE /claude-code/plugins/:id`

If those endpoints return `404`, the existing `/v1/skills` paths remain the fallback. This preserves current LiteLLM Skills Gateway installs while supporting the newer Skill Hub surface.

## Verification

- Start from the current `npm run check` green baseline.
- Add focused failing tests first:
  - `tests/discover.test.ts` for response-mode model inclusion and API override.
  - `tests/mcp-tools.test.ts` for retry and parallel execution metadata.
  - `tests/skills.test.ts` for Skill Hub fetch/create/delete plus fallback behavior.
- Run focused test files until green.
- Run `npm run check`.
- Run `npm run clean && npm run build` because runtime source changes affect `dist`.
