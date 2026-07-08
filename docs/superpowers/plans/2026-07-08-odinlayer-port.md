# OdinLayer Behavior Port Plan

## Goal

Port the approved useful OdinLayer behaviors into the current `main` branch with minimal source changes and no new runtime dependencies.

## Architecture

Discovery maps LiteLLM model metadata into Pi model configs. MCP tools translate LiteLLM MCP REST tools into Pi tools. Skills translate LiteLLM skill/plugin endpoints into prompt context and management tools. Each port stays in the owning module.

## Tech Stack

TypeScript ESM, Vitest, existing Pi extension APIs.

## Constraints

- Do not add runtime dependencies.
- Do not edit `dist/` by hand.
- Keep `/litellm-refresh` unchanged.
- Preserve `/v1/skills` compatibility.
- Use TDD: write the focused failing test first, then implementation.

## Task 1: Responses API Discovery

**Files:**

- Modify: `src/discover.ts`
- Test: `tests/discover.test.ts`

**Interfaces:**

- Consumes `ModelInfoEntry.model_info.mode`.
- Produces `ProviderModelConfig.api = "openai-responses"` only for `response` or `responses` modes.

**Steps:**

1. Add failing tests for `/model/info` response-mode models and `/health` fallback response-mode models.
2. Run `npm test -- tests/discover.test.ts`.
3. Update discovery mode filtering and model mapping.
4. Run `npm test -- tests/discover.test.ts`.

## Task 2: MCP Retry And Parallel Execution

**Files:**

- Modify: `src/mcp-tools.ts`
- Test: `tests/mcp-tools.test.ts`

**Interfaces:**

- Generated MCP tool definitions expose `executionMode: "parallel"`.
- `executeMcpTool()` retries once for retryable HTTP/network failures.

**Steps:**

1. Add failing tests for `executionMode`, retryable HTTP retry, and non-retryable HTTP no-retry.
2. Run `npm test -- tests/mcp-tools.test.ts`.
3. Add one retry attempt in `executeMcpTool()` and set tool execution mode.
4. Run `npm test -- tests/mcp-tools.test.ts`.

## Task 3: Skill Hub Compatibility

**Files:**

- Modify: `src/skills.ts`
- Modify: `src/types.ts`
- Test: `tests/skills.test.ts`
- Modify: `README.md`

**Interfaces:**

- List from `GET /claude-code/marketplace.json`, falling back to `GET /v1/skills`.
- Create at `POST /claude-code/plugins`, falling back to `POST /v1/skills` on `404`.
- Delete at `DELETE /claude-code/plugins/:id`, falling back to `DELETE /v1/skills/:id` on `404`.

**Steps:**

1. Add failing tests for Skill Hub list/create/delete and `/v1/skills` fallback.
2. Run `npm test -- tests/skills.test.ts`.
3. Update skill body normalization and endpoint fallback logic.
4. Update README wording for Skill Hub plus Skills Gateway compatibility.
5. Run `npm test -- tests/skills.test.ts`.

## Final Verification

1. Run `npm run check`.
2. Run `npm run clean && npm run build`.
3. Commit each logical change locally with a conventional commit message.
