# Agent Notes

## Project Shape

- This package is a Pi extension that registers a `litellm` provider from `src/index.ts`.
- Source is TypeScript ESM under `src/`; tests are Vitest specs under `tests/`.
- Build output is `dist/`; do not edit generated output by hand or publish it.
- Git and npm installs load `./src/index.ts` through `package.json` `pi.extensions`.
- Node support starts at `>=22.19.0`; GitHub workflows currently run Node `26.5.0`.
- Dev dependencies track Pi `0.99.1` while `peerDependencies` stay `>=0.83.0`. Use newer Pi APIs only in ways older Pi
  ignores (extra `ToolDefinition` fields) or feature-detects (an `exposure` field on `pi.getAllTools()` entries), and do
  not import runtime symbols that older `pi-ai`/`pi-coding-agent` lack, such as `isModelType`.
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

## LiteLLM MCP Tools

- Everything a LiteLLM MCP server returns is untrusted input: tool names, server names, descriptions, schemas, and results.
- Refuse `pattern`/`patternProperties` at a keyword position regardless of value type. The current TypeBox only compiles a string-valued `pattern`, but the expression text would still reach Pi and the provider, and the guard should not rest on an upstream type check.
- The safety invariant is that **no proxy-supplied regular expression, and no reference the validator cannot safely resolve, reaches Pi or TypeBox**. `pattern` and `patternProperties` keys are compiled into backtracking regexes with no time limit; `$ref` is resolved as an arbitrary JSON pointer and whatever it lands on is then treated as a schema.
- Do not implement that check by enumerating the schema positions where a subschema may appear. A `$ref` resolves an arbitrary JSON pointer, so a regex can hide under any key, including data-only ones like `default` and `examples`. `findSchemaHazard()` in `src/mcp-tools.ts` walks the entire supplied graph instead, bounded by depth, a node budget, and identity-based cycle detection; an incomplete walk is a hazard, not a pass.
- Covering the graph is necessary but not sufficient for references: a local pointer must also *resolve* to a usable subschema, and the reference chain must be proven acyclic. Object identity cannot see a pointer cycle, because each hop is a different object.
- Test the guard by exercising the consumer, not just by asserting a marker string is absent. `tests/mcp-tools.test.ts` compiles every registered schema with the same TypeBox the runtime uses and watches the `RegExp` constructor, covering executable expressions that do not contain an expected marker string.
- A hazard degrades the tool to the extension-owned `args` envelope; it does not drop the tool. Real MCP schemas commonly use `pattern`, so dropping them costs real functionality. The envelope's own library-generated `patternProperties` is fine because it is not proxy-supplied.
- A present-but-non-object `inputSchema` is `invalid-schema`, never treated as schemaless. Only an absent or empty schema takes the envelope as its normal path.
- Tool names must stay a pure function of tool identity. Always append the identity hash; never make the hash conditional on what else is in the catalog, or adding a sibling will rename a survivor and Pi cannot unregister the old name.
- Every raw catalog entry must land in exactly one count or class, so `raw = prepared + dropped + beyond-the-cap` always reconciles. A registration refusal is reported separately because registered may be less than prepared. Add a class rather than letting a loss go unreported.
- Diagnostics dedupe on full membership, not on the printed sample, and a class that stops occurring is cleared. Never interpolate proxy-supplied text or credentials into stderr; use generated names for normalized tools and positional labels for entries that failed normalization, which may have no usable name. Derive malformed-entry identities with queued work bounded by nesting depth; retain at most one key list per open object rather than one work item per child.
- The MCP catalog identity must not hold credential material. `credentialFingerprint()` reduces the API key **and** the headers (`LITELLM_HEADERS` can carry its own authorization) to a per-process salted HMAC, so a change to either still forces re-registration while nothing reversible is retained. Note `src/gcloud-token.ts` builds its cache key from the raw refresh token — same defect class, untouched baseline, out of scope here.
- `pi.registerTool` is a synchronous replace-by-name whose only failure is a never-reset staleness check, so a refusal is fatal for the pass and for that extension instance. Do not model it as a per-tool rejection or add retry; a reload provides a clean instance.
- Tools cannot be unregistered, so a generated name missing from the next successful, non-partial pass is re-registered as a `hidden` stub whose `execute` fails with extension-owned text (never a proxy call, since older Pi ignores `exposure`). It leaves the active set only when Pi honors exposure (`piHonorsToolExposure()`).
- `POST /mcp-rest/tools/call` is side-effecting and must stay exactly-once, and cancellation must preserve the caller's original abort reason. Keep the tests that assert call counts and reason identity.
- Two upstream behaviors keep passthrough schemas safe and are not controlled here. Both are pinned by tests in `tests/mcp-tools.test.ts`; if a dependency bump breaks either, fix it there before shipping.
  - `typebox`'s `value/convert/from_object.mjs` turns `properties` keys into `new RegExp(`^${key}$`)` with **no escaping**, and `pi-ai` calls `Value.Convert` on tool parameters. That is only harmless because `Convert` walks recognised TypeBox types and no-ops on a raw JSON Schema, so a proxy-supplied property name never reaches it. If that changed, a property named `(a+)+$` would become an executable backtracking regex tested against model-supplied argument keys.
  - `format` is live on passthrough schemas: it is a proxy-chosen selector of `typebox`'s own regexes, executed against model-supplied strings. The shipped formats are well-anchored, so this is a residual dependency on upstream regex quality, not a hole. Do not assume `format` is ignored.
- Do not write timing-based tests for any of this. Assert the registered `parameters` and the absence of the exact proxy-supplied regex or ref, and keep the schema-position test lists independent of the implementation's own tables.
- `litellm.mcp.exposure` and `toolExposure` mirror Pi's `mcp.json` keys and are parsed once by `parseMcpExposurePolicy()`. `codemode-deferred` becomes Pi's `deferred` on the tool definition, while `report.exposures` keeps the MCP value so activation can tell them apart. Activation mirrors the built-in MCP extension's `ensureDiscoveryActive`: codemode for `codemode`/`codemode-deferred`, `tool_search` for `deferred`, one warning when neither is registered, and one warning when `getAllTools()` entries carry no `exposure` (Pi before 0.99, where every tool is direct).
- Never register `/mcp`, `codemode`, or `tool_search`: Pi 0.99 unloads the built-in extension whose tool, command, or flag an extension re-registers. Pi's own MCP tools are `mcp__<server>__<tool>`, so `mcp_<server>_<tool>_<hash>` never collides with them.
- Annotations keep only the four boolean hints, as Pi's own MCP client does; `title` and any other proxy-supplied annotation field never reach Pi. Namespace names share the generated tool-name prefix (`mcp_<server>` or `mcp_<alias>_<server>`) and are bounded like tool names; the namespace description is proxy text bounded to 256 bytes.

## Reasoning Policy

- Discovered `litellmPolicy` scopes request and response behavior to backend evidence; route text never authorizes generation controls or request-side visibility parameters.
- Share bounded backend identity parsing across catalog, family, and generation decisions. Generic adapter labels do not override an identified backend vendor; custom Azure authority wins over a generic OpenAI adapter.
- Every level in `THINKING_LEVEL_DEFINITIONS` has a LiteLLM support flag, including `medium` and `high`; LiteLLM 1.86 emits `supports_medium_reasoning_effort` and `supports_high_reasoning_effort`. Derive the flag table from that one definition rather than restating it, or a level goes unread.
- An absent `supported_openai_params` list plus explicit `supports_reasoning: true` is an operator opt-in to the `reasoning_effort` carrier (LiteLLM omits the list for deployments outside its model map). A declared list without the carrier still denies, and Kimi/DeepSeek families never take the opt-in.
- A declared `reasoning_effort_levels` list answers every level for its deployment, ahead of the per-level flags.
- Catalog level maps are tristate, not complete lists: an omitted standard level keeps Pi's default.
- Null/absent flags have no opinion; explicit denials win and extended effort levels need explicit support. A router flag is the more specific evidence, so an explicit `true` grants a level over a catalog map that denies it.
- Close thinking levels against the protocol and accepted carrier actually used after wildcard expansion. Responses compatibility contains only Responses fields.

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
