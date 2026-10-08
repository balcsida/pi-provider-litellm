# Agent Notes

## Project Shape

- This package is a Pi extension that registers a `litellm` provider from `src/index.ts`.
- Source is TypeScript ESM under `src/`; tests are Vitest specs under `tests/`.
- Build output is `dist/`; do not edit generated output by hand or publish it.
- Git and npm installs load `./src/index.ts` through `package.json` `pi.extensions`.
- Node support starts at `>=22.19.0`; GitHub workflows currently run Node `26.5.0`.
- Dev dependencies track Pi `1.0.3` while `peerDependencies` stay `>=0.99.2`. Pi installs packages without resolving
  peers, so the `VERSION` check at the top of the extension factory enforces that floor (read through a namespace
  import, so a Pi without `VERSION` still reaches it); raise both together. Use newer Pi APIs only in ways Pi 0.99.2
  ignores or feature-detects, and do not import runtime symbols its `pi-ai` or `pi-coding-agent` lacks.
- Pi 0.99 persists models of every type in `models-store.json`; `refreshModels` narrows stored models to chat models
  before reading chat-only fields.

## Commands

- Use `npm ci` when reinstalling dependencies from the lockfile.
- Use `npm test` for the full test suite.
- Use `npm test -- tests/<file>.test.ts` for a focused Vitest run.
- Use `npm run probe:proxy -- --snapshot <fixture> [--src <worktree>]` for snapshot probing, or `npm run probe:proxy -- --live --base-url $LITELLM_DEV_BASE_URL --max-requests 100` for the bounded live matrix. Live Chat probes pass through the same discovered request policy as Pi.
- Use `npm run check` before committing code changes; it runs Biome, typecheck, tests, and the package-content guard.
- Use `npm run clean && npm run build` when changing exported/runtime code.
- Use `npm run supply-chain:guard` and `npm pack --dry-run` when package contents, dependency policy, or release packaging change.

## Discovery And Credentials

- Model discovery lives in `src/discover.ts`; pure deployment-group reduction lives in `src/model-groups.ts`.
- Shared thinking-level definitions and map intersection live in `src/thinking-levels.ts`.
- Prefer `/model/info` for rich metadata. Use `/v1/models` as the status-code fallback only when `/model/info`
  returns 401, 403, or 404; after a successful `/model/info`, also query `/v1/models` only to expand wildcard routes.
- Treat `model_name` as a public route group, not backend evidence. Resolve backend identity from
  `model_info.base_model` before `litellm_params.model`; treat `custom_llm_provider` as provider evidence, withholding
  identity when a non-generic provider conflicts with the model prefix. Reduce every deployment before choosing
  transport, capabilities, limits, prices, or catalog authority; never shallow-merge duplicate route rows.
- Keep catalog lookup provider-aware. Unqualified or conflicting identities must not scan every Pi provider catalog.
- `supported_endpoints` pins transport except for `azure_ai` deployments on a proxy older than LiteLLM v1.103.0 or of
  unknown version. Those releases bridge them to Chat Completions regardless of that cost-map list, so they take
  Responses only from an explicit `mode: "responses"`. Keep `scripts/probe-proxy.ts` `protocolPrediction` in step
  with `supportsResponses` in `src/discover.ts`.
- Version-gated workarounds read the proxy version through `src/proxy-version.ts`. An unknown version satisfies no
  floor, so every gate fails toward the workaround, and a pre-release orders before the release it names. Probe only
  when a published deployment's outcome depends on the version (`publishesVersionGatedTransport`), at most once per
  discovery: the probe draws an error reply, so an unconditional one would add a failed request to every operator's
  logs. A listed deployment is not yet a published one: a withheld group publishes nothing, and a wildcard route
  publishes nothing until `/v1/models` expands it, so its probe waits for the expansion. `scripts/probe-proxy.ts`
  reuses `DiscoveryResult.proxyVersion` and never probes. The version is proxy-supplied; keep only the parsed numbers
  and never print the raw header.
- Evidence-free `/v1/models` and wholly health-only `/health` groups take protocol and presentation metadata from
  the bounded Pi catalog lookup. `openai-responses` selects Responses; other or missing catalog APIs select Chat.
  A health-only row mixed with deployment details grants no catalog authority to that group.
- ` (no metadata)` is the evidence-free fallback cache enrichment marker. Reduced `/model/info` groups and health
  groups with deployment details use ` (incomplete metadata)`, which remains ineligible for route-name enrichment.
- Keep `LITELLM_OFFLINE` and `LITELLM_DISCOVERY_TIMEOUT_MS` behavior compatible with README docs.
- `/litellm-refresh` is the explicit refresh that Pi's `PI_OFFLINE` otherwise blocks. It passes `allowNetwork: true`
  but keeps the `LITELLM_OFFLINE=1` and zero-timeout gates, and models.dev stays off under `PI_OFFLINE`; do not widen it
  past the configured proxies.
- Stored Pi `/login litellm` credentials take precedence over `LITELLM_API_KEY`.
- Bump `LITELLM_DISCOVERY_VERSION` (and its pin in `tests/backend-identity.test.ts`) whenever discovery adds or changes persisted model metadata such as a `litellmPolicy` field. Pi keeps a same-version stored entry over what startup discovery just proved, so older caches would otherwise mask the change.
- Pi stores discovered models in `models-store.json`; models.dev enrichment is opt-in with `LITELLM_MODELS_DEV=1` and
  uses `litellm-models-dev.json` with a 28-day cache window under the Pi agent dir. When it is off, discovery must not
  read that cache either.
- Pi owns discovered-model persistence in `models-store.json`; this extension does not write a model cache. Legacy `litellm-models.json` model caches are ignored and never deleted. `litellm-models-dev.json` is the models.dev cache and is refreshed in place.
- Direct OIDC login (`litellm.providers.litellm.oidc`, global settings only; persisted flow `oidc_pkce`) talks only to
  the configured IdP: it sends nothing to the proxy, never derives the IdP from proxy metadata, never sends
  `LITELLM_HEADERS` or provider headers to the IdP, uses `redirect: "manual"` with any 3xx a failure, and reports only
  an OAuth error code matching `^[a-z_]{1,64}$` or an HTTP status. Its refresh branch must stay ahead of the `!command`
  fallthrough so an IdP refresh token is never executed. The proxy, not Pi, verifies the id_token signature.
- Google ADC is resolved in process through `src/gcloud-token.ts`; there is no helper subprocess and no `src/gcloud-token-cli.ts`. Only `authorized_user` credentials are supported, and service accounts warn and fail closed.
- `apiKey.check` performs no network call, so it reports credential shape, not mintability. Its source label must mirror the precedence in `resolveCredentials`, so ADC is named whenever a complete ADC file exists; if the refresh token no longer mints, `resolve` falls back and reports the credential it actually used.

## LiteLLM Request Hooks

- `before_provider_request` is a global Pi hook. Only mutate provider payloads when `ctx.model?.provider` matches the default `litellm` provider or a registered alias from `litellm.providers`.
- Do not add user-facing flags or environment variables to hide provider-scoping bugs.
- `before_provider_headers` sends Pi's canonical session id as `x-litellm-session-id`, scoped to the configured LiteLLM provider names; no session field is added to request bodies.
- Kimi/Moonshot responses may include `<think>` text; Pi-visible normalization in `message_end` follows discovered response policy. Request-side visibility suppression additionally requires every deployment's routing provider to be Moonshot. Keep hosted Kimi paths covered by feature tests.

## LiteLLM MCP

- MCP goes through Pi's built-in client. Each provider registers `<root>/mcp` with
  `pi.registerMcpServer(name, { url, auth: { provider: name }, headers?, exposure?, toolExposure? })`. Do not
  reimplement tool discovery, naming, exposure, schema handling, or calls here.
- The registration never copies the provider's key. `auth.provider` makes Pi read the provider's current credential on
  every request, which is what keeps OAuth refresh, ADC, and key helpers working. Never put the key in `headers`; it
  would be frozen at registration. Custom headers are already resolved, so `piLiteral()` escapes `$` and a leading `!`
  before Pi's own `${VAR}`/`!cmd` resolution sees them.
- Because the token is read per request, a connection must never outlive its proxy root: `/login` drops the server when
  it starts, before the new credential is stored, and every replacement unregisters the old server first.
- Pass `exposure` and `toolExposure` through unparsed. Pi validates them, and the error `registerMcpServer` throws is
  the report; do not add defaults or a second validator.
- The remembered registration identity is an HMAC of the config, because custom headers can carry credentials. Register
  again only when it changes, unregister when no proxy root remains (logout), and register nothing under
  `LITELLM_OFFLINE`, a zero discovery timeout, or `PI_OFFLINE`.
- Never register `/mcp`, `codemode`, or `tool_search`: Pi unloads the built-in extension whose tool, command, or flag an
  extension re-registers.

## Budget Status

- `src/budget.ts` hooks fire only for the configured LiteLLM providers and are registered after `setupLiteLLMCostTracking`, because `tests/features.test.ts` calls `after_provider_response` handler `[0]` and expects the cost hook.
- Automatic polls run only with a UI and never under `LITELLM_OFFLINE=1`, a zero discovery timeout, or `PI_OFFLINE`; `/litellm-budget` ignores only `PI_OFFLINE`.
- The footer and command show no proxy-supplied text (aliases, messages), and responses are never logged. The credential
  and custom headers are kept only as an HMAC digest under a per-session random key, like the MCP registration identity;
  CodeQL's `js/insufficient-password-hash` flags it, a false positive for an in-memory change detector.
- A 4xx (not 429) or 500 endpoint is not retried until the credential digest (key or custom headers) changes or the
  command runs. LiteLLM's 500
  here repeats on every call (no database, a credential it cannot look up), while 429 and 502-504 stay retryable.
- Polls set the key level; headers only raise it, or set it alone while `/key/info` has not succeeded.
- A JWT credential (three dot-separated parts, LiteLLM's own `is_jwt` test), such as a Direct OIDC login's, has no key
  row (LiteLLM 1.102 answers `/key/info` with 500); its team, member, and org come from the user's only team in
  `/v2/user/info` (`/user/info`). A user in several teams gets the user level only: which team LiteLLM charges depends
  on the proxy's JWT settings. Never apply this to a virtual key whose `/key/info` is denied: its user's team need not
  be the one it is charged to (the master key's default user is added to every team the master key creates).

## Reasoning Policy

- Discovered `litellmPolicy` scopes request and response behavior to backend evidence; route text never authorizes generation controls or request-side visibility parameters.
- Share bounded backend identity parsing across catalog, family, and generation decisions. Generic adapter labels do not override an identified backend vendor; custom Azure authority wins over a generic OpenAI adapter.
- Every level in `THINKING_LEVEL_DEFINITIONS` has a LiteLLM support flag, including `medium` and `high`; LiteLLM 1.86 emits `supports_medium_reasoning_effort` and `supports_high_reasoning_effort`. Derive the flag table from that one definition rather than restating it, or a level goes unread.
- An absent `supported_openai_params` list plus explicit `supports_reasoning: true` is an operator opt-in to the `reasoning_effort` carrier (LiteLLM omits the list for deployments outside its model map). A declared list without the carrier still denies, and Kimi/DeepSeek families never take the opt-in.
- A declared `reasoning_effort_levels` list answers every level for its deployment, ahead of the per-level flags.
- Catalog level maps are tristate, not complete lists: an omitted standard level keeps Pi's default.
- Null/absent flags have no opinion; explicit denials win and extended effort levels need explicit support. A router flag is the more specific evidence, so an explicit `true` grants a level over a catalog map that denies it.
- Close thinking levels against the protocol and accepted carrier actually used after wildcard expansion. Responses compatibility contains only Responses fields.
- GPT-5.5 and later reject `reasoning_effort` with function tools on Chat Completions. Discovery sets `litellmPolicy.dropToolReasoning` for Chat routes when any deployment's backend id (never the route name) is GPT-5.5+, and wildcard combinators take any parent's flag: dropping reasoning only degrades a request, keeping it fails one. The request hook uses the route name only for a model without `litellmBackendFamily` evidence. It sends the model's declared `thinkingLevelMap.off` effort, and without one omits the field, except under `explicitToolReasoningOff` (GPT-6+ backend evidence, or the route name without evidence), which sends `none` because GPT-6 falls back to its default effort when the field is omitted. `off: null` from `NO_TRANSMISSIBLE_LEVELS` means no carrier evidence, not a model that cannot turn reasoning off.

## Compatibility Rules

- Provider-specific request compatibility belongs in discovered model `compat` metadata, not broad runtime mutation.
- Native Messages requires unanimous compatible Claude deployment evidence; evidence-free fallback and health discovery never select it. Keep Messages compatibility separate from Chat and Responses fields.
- Strict tools require every deployment's declared routing (`custom_llm_provider` and the `litellm_params.model` prefix) to name Anthropic. `model_info.litellm_provider` is a cost-map lookup of `base_model`, not routing, and `supports_response_schema` / `supports_native_structured_output` describe JSON output, not tool definitions; neither is strict-tool evidence.
- Chat Completions `strict` tool schemas: since Pi 0.87 the `openai-completions` default for unknown endpoints is
  `supportsStrictMode: false`, so LiteLLM Chat routes send no `strict` field. Pi only ever sent `strict: true` for tools
  that opt in through `constrainedSampling`, which nothing here uses, so this is left at Pi's fail-closed default;
  revisit only if such a tool needs strict enforcement on OpenAI/Azure-routed deployments.
- Kimi/Moonshot-style compatibility is split across `completionsCompat()` and `responsesCompat()`; `buildCompat()` is retained only as the completions alias. Keep regression tests with model discovery changes.
- Anthropic-backed aliases using `openai-completions` need `cacheControlFormat: "anthropic"` so Pi forwards prompt-cache markers through LiteLLM; `openai-responses` uses its native prompt cache fields instead.

## Smoke And CI

- CI runs `npm ci` and `npm run prepublishOnly`; release relies on `npm publish` invoking `prepublishOnly`.
- `.github/workflows/litellm-smoke.yml` uses VidaiMock plus a real LiteLLM proxy; it should not require real provider API keys.
- Keep smoke readiness probes bounded with `curl --connect-timeout 1 --max-time 3`.
- `scripts/smoke-runner.ts` exercises discovery and all three selected protocol endpoints through the proxy; the Pi CLI smoke independently proves the extension's native Messages path.
- The non-interactive Pi CLI smoke loads the package root (`-e .`) so it exercises `pi.extensions` resolution; the interactive terminal smoke loads `src/index.ts` by path.
- `--list-models` alone does not prove an extension loaded, because Pi also reports models from its own store. Assert a load-specific side effect instead.

## Release And Packaging

- The release workflow is tag-driven for `v*.*.*`; it publishes with `npm publish --access public --provenance` and creates a GitHub release.
- Local release prep should keep `package.json` and `package-lock.json` versions in sync, build `dist/`, run package checks, and create only local commits/tags unless the user explicitly overrides the no-push rule.
- Verify released state with `gh release view <tag>` and `npm view pi-provider-litellm version dist-tags --json` after the user pushes the tag.
- The npm package should stay limited to `src`, `README.md`, and `LICENSE`; builds are verification-only.
- `scripts/supply-chain-guard.ts` rejects install lifecycle scripts, runtime dependencies, non-registry specs, non-registry lockfile URLs, and unexpected package files, and requires every allowlisted source file to ship; update tests before changing that policy.

## Package Metadata

- Keep the Pi gallery image URL in `package.json` exactly as declared unless the user asks to change it.
- Do not include gallery assets in the npm package unless explicitly requested; verify packaging with `npm pack --dry-run` when package contents change.
