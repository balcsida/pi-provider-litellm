# Agent Notes

## Project Shape

- This package is a Pi extension that registers a `litellm` provider from `src/index.ts`.
- Source is TypeScript ESM under `src/`; tests are Vitest specs under `tests/`.
- Build output is `dist/`; do not edit generated output by hand.
- The package entrypoint is `./dist/index.js`, and the Pi extension registration comes from `package.json` `pi.extensions`.

## Commands

- Use `npm test` for the full test suite.
- Use `npm test -- tests/<file>.test.ts` for a focused Vitest run.
- Use `npm run check` before committing code changes; it runs Biome, typecheck, and tests.
- Use `npm run build` when changing exported/runtime code.

## Discovery And Credentials

- Model discovery lives in `src/discover.ts`.
- Prefer `/model/info` for rich metadata; fallback to `/v1/models` only on 401, 403, or 404.
- Keep `LITELLM_OFFLINE` and `LITELLM_DISCOVERY_TIMEOUT_MS` behavior compatible with README docs.
- Stored Pi `/login litellm` credentials take precedence over `LITELLM_API_KEY`.

## LiteLLM Request Hooks

- `before_provider_request` is a global Pi hook. Only mutate provider payloads when `ctx.model?.provider === "litellm"`.
- Do not add user-facing flags or environment variables to hide provider-scoping bugs.
- `litellm_session_id` is optional LiteLLM session grouping metadata. If a LiteLLM server rejects it for LiteLLM-routed requests, keep Pi requests working first and document the admin-facing recommendation separately.

## Compatibility Rules

- Provider-specific request compatibility belongs in discovered model `compat` metadata, not broad runtime mutation.
- Kimi/Moonshot-style models are handled in `buildCompat()`; keep regression tests with model discovery changes.
- Anthropic-backed aliases need `cacheControlFormat: "anthropic"` so Pi forwards prompt-cache markers through LiteLLM.

## Package Metadata

- Keep the Pi gallery image URL in `package.json` exactly as declared unless the user asks to change it.
- Do not include gallery assets in the npm package unless explicitly requested; verify packaging with `npm pack --dry-run` when package contents change.
