# Porting pi-provider-litellm to PiG (Go)

This directory holds the Go port of the TypeScript extension in `../src`. The TypeScript source is the
specification: port observed behavior, keep identifiers recognizable (camelCase → PascalCase), and cite the
source file in a `// Ports src/<file>.ts` comment at the top of each Go file.

## Layout

```text
pig/
├── PORTING.md                      this brief
├── package.json                    PiG package manifest (phase 6)
└── extensions/litellm/             Go module github.com/balcsida/pi-provider-litellm/pig/extensions/litellm
    ├── go.mod                      requires github.com/MichaelKinsy/PiG v0.5.2 and .../extensions/sdk v0.5.2
    ├── extension.go                factory: func Extension() *sdk.Extension      ← src/index.ts
    ├── provider.go                 LiteLLM native provider                       ← src/provider.ts
    ├── models_store.go             stored-catalog seed                           ← src/models-store.ts
    ├── auth.go, oauth.go           credential resolution, login flows            ← src/index.ts
    ├── request_policy.go           request/response hooks                        ← src/index.ts, src/cost.ts
    ├── budget.go, skills.go, mcp.go, gcloud.go, settings.go                      ← src/budget.ts, skills.ts, index.ts, gcloud-token.ts
    └── internal/
        ├── types/                  ← src/types.ts
        ├── backend/                ← src/backend-identity.ts
        ├── thinking/               ← src/thinking-levels.ts
        ├── proxyversion/           ← src/proxy-version.ts
        ├── protocols/              ← src/protocols.ts (+ normalizeBaseUrl from src/discover.ts)
        ├── modelgroups/            ← src/model-groups.ts
        ├── catalog/                ← src/public-catalog.ts (reads PiG's ai catalog instead of pi-ai's)
        ├── discover/               ← src/discover.ts
        ├── bridge/                 ai.ModelsProvider ⇄ sdk.Provider adapter (new; PiG has no equivalent)
        └── fixtures/               test helper locating ../../tests/fixtures
```

The extension identity is `litellm` (PiG derives it from the directory name). The provider id stays `litellm`,
aliases come from settings and from the multi-provider environment variables, as in the TypeScript version.

## Design decisions

- **Native provider.** `ai.CreateProvider` (PiG's port of pi-ai `createProvider`) with an `ai.ProviderAPIMap`
  holding PiG's built-in `anthropic-messages`, `openai-completions` and `openai-responses` drivers, wrapped
  exactly as `src/provider.ts` wraps pi-ai: `fetchModels`, `filterModels`, `refreshModels`, and the
  `requestModel` guard around `stream`/`streamSimple`. `internal/bridge` turns the `*ai.ModelsProvider` into the
  wire-shaped `*sdk.Provider` that `sdk.Extension.RegisterNativeProvider` takes.
- **Model JSON shape.** Models cross the SDK wire as JSON objects. The authoritative shape is
  `modelsCatalogRecord` in PiG `ai/models_catalog_codec.go` (`id`, `name`, `api`, `provider`, `baseUrl`,
  `reasoning`, `thinkingLevelMap`, `input`, `inputLimits`, `cost{input,output,cacheRead,cacheWrite,tiers}`,
  `promptCache`, `contextWindow`, `maxTokens`, `samplingParams`, `samplingParamsByThinkingLevel`, `headers`,
  `compat`). PiG decodes into a closed struct, so the extension-specific fields the TypeScript version puts on
  the model (`litellmPolicy`, `litellmBackendFamily`, `litellmResponsesReasoningControl`,
  `suppressReasoningContent`, `litellmDiscoveryVersion`) do **not** survive a round trip through the host.
  Discovery still produces them, but the extension keeps them in its own in-memory map keyed by
  `provider + "\x00" + model id`, persisted in a sidecar JSON file next to PiG's model store, and hooks look
  them up there instead of reading them off `ctx.Model`.
- **Streaming belongs to PiG.** No protocol code is written here. `ai.StreamSimple`/the drivers in PiG `ai`
  do Chat Completions, Responses and Messages.
- **Everything else is a 1:1 port.** Discovery, deployment-group reduction, backend identity, thinking levels,
  proxy version gates, request/response policies, budget, cost, skills, MCP registration, credentials and the
  three login flows keep the TypeScript behavior and its tests.
- **Prompt cache lifetimes.** Discovery sets `promptCache` (`short` 300, `long` 3600) exactly like the TypeScript
  rule: unanimous Claude backend-family evidence in the `/model/info` mapper, on `anthropic-messages` or on
  `openai-completions` with `cacheControlFormat == "anthropic"`; never on Responses, the `/v1/models` fallback or a
  route identified only by its name. GPT-5.6 and later get `short` 1800 alone, again like the TypeScript rule: every
  deployment's backend identity must be OpenAI-family GPT-5.6+, routed only through `openai`, `azure` or `azure_ai`
  (`backend.RoutesOnlyThrough`), on `openai-responses` only. A wildcard child takes its selected parent's value,
  reduced from the backend row its id substitutes into. `promptCache` is part of `modelsCatalogRecord`, so it crosses
  the SDK wire and needs no sidecar entry. `DiscoveryVersion` is 10, in step with `LITELLM_DISCOVERY_VERSION` (the Go
  side stayed at 7 through TypeScript's 8, which only added the classifier routes it withholds).

## Conventions

- Go 1.26, standard library plus the two PiG modules already in `go.mod`. Do not add dependencies or edit
  `go.mod`/`go.sum`; report a need instead.
- Fuse-compatible code only: no `fmt.Print*`, `os.Stdout`, `os.Exit`, `os.Chdir`, `log.Fatal*`. Report through
  `sdk.Context` (`Notify`, `SetStatus`) or returned errors. Diagnostics the TypeScript version writes to stderr
  go to `os.Stderr` only through one helper in the root package, never `os.Stdout`.
- `AbortSignal` → `context.Context`. An awaited Promise is a blocking call. No detached goroutines without an
  owner that cancels, reports errors and drains on shutdown (budget polling is the one timer-driven exception).
- JSON: use structs with tags for known shapes and `map[string]any` only where the SDK wire requires it.
  Distinguish absent, `null` and `false` exactly where the TypeScript code does (pointer fields).
- Tests are Go table tests ported from `../tests/*.test.ts`, named after the Vitest `describe`/`it` they port.
  They read the shared fixtures through `internal/fixtures` (`fixtures.Path(t, "proxy/prod-model-info-2026-09-04.json")`).
  Network tests use `httptest`. Never contact a real proxy or read credentials.
- `gofmt -l`, `go vet ./...` and `go test ./...` must be clean before a task is reported done.
- Commits follow Conventional Commits with scope `pig`, e.g. `feat(pig): port backend identity`.

## Environment and settings compatibility

Keep every `LITELLM_*` variable, the `litellm` settings block (`providers`, `mcp`, `budget`, `skills`) and the
`auth.json` entry shapes (`oauth` with `access`/`refresh`/`expires`/`baseUrl`, `api_key` with `key`) exactly as
the README and `src/index.ts` define them, so one configuration works for both runtimes. PiG's agent directory
replaces Pi's (`~/.pig/agent`): resolve it through the SDK, never hard-code it.

Provider config may also live in the agent dir's `models.json`, read as JSONC the way PiG reads it (BOM, `//`
comments, trailing commas; `stripJSONComments` is copied from PiG's internal `model_registry.go`). Only providers
are taken from it: the native `providers.litellm` slot and `litellm.providers.<name>`, merged per field over
`settings.json` with `models.json` winning. A leading BOM is also accepted in `settings.json`.

## Verification ladder

1. `go test ./...` in `pig/extensions/litellm`.
2. `pig install --validate-only --json ./pig/extensions/litellm` proves the factory builds and registers: PiG 0.5.2
   reports `"valid": true` and `"registered": true`. `pig package validate ./pig --json` validates the Go member
   too.
3. Live smoke against the mock proxy in `pig/tools/mockproxy` (`go run . -key test-key -addr 127.0.0.1:48417`):

   ```sh
   export PIG_CODING_AGENT_DIR=$(mktemp -d) LITELLM_BASE_URL=http://127.0.0.1:48417 LITELLM_API_KEY=test-key
   pig -e ./pig/extensions/litellm --list-models            # lists mock-chat, mock-claude, mock-responses
   pig -e ./pig/extensions/litellm --model litellm/mock-chat -p "say hi"   # streams "mock reply to: ..."
   ```

   Repeat the prompt with `mock-responses` and `mock-claude`: the mock logs `POST /v1/responses` and
   `POST /v1/messages`, which proves each model reached its own protocol's endpoint.

   Run the same steps once more with no `LITELLM_*` variables and only
   `{"providers":{"litellm":{"baseUrl":"http://127.0.0.1:48417","apiKey":"test-key"}}}` in
   `$PIG_CODING_AGENT_DIR/models.json`. PiG composes that block over the registered provider itself, so this proves
   its composition keeps each model's route; it stands in for the Vitest case that drives Pi's real ModelRuntime.

   To measure startup against a slow proxy, start the mock with `-latency 300ms` and compare the wall time of
   `pig -e ./pig/extensions/litellm --model litellm/mock-chat -p "say hi"` between two builds. `PIG_STARTUP_TRACE=1`
   prints the host's startup steps, but `session_start` is not one of them, so wall time is the number to compare.

4. `pig piglet build pig/litellm-example.yaml --format binary --out /tmp/pig-litellm-bin` fuses the extension into a
   PiG binary; the build's process-hazard vet must pass, and the binary must pass step 3 unchanged.
5. Login inside a session: `pig/tools/login-smoke/login-smoke.mts` drives `/login litellm` through a real terminal
   against the mock proxy's SSO modes (`go run . -key test-key -addr 127.0.0.1:48417 -sso cli|pkce|paste`), then
   proves the stored `auth.json` entry lists models and chats outside the TUI, and for PKCE that an expired entry is
   refreshed through PiG's host:

   ```sh
   PIG_BIN=pig LITELLM_BASE_URL=http://127.0.0.1:48417 LOGIN_SMOKE_MODE=cli npx tsx pig/tools/login-smoke/login-smoke.mts
   ```

   `LOGIN_SMOKE_MODE` is `cli`, `pkce`, `paste` or `apikey`; the mock's `-sso` mode must match the first three.
   Stock PiG 0.5.2 passes this step in all four modes. On a desktop PiG also opens the authorize URL in the browser;
   when the browser's callback lands first, the `pkce` driver's own fetch fails with `ECONNREFUSED`, so run it again.

## Status

All six phases are ported: discovery and model-group reduction, the native provider with the request guard and
refresh wrapper, credentials and provider auth, request and response policy hooks with cost tracking, the login
flows (CLI PKCE, direct OIDC, CLI SSO, pasted token) with refresh, budget status, Skills tools, MCP registration,
the PiG package manifest, an example fused Piglet and a CI workflow. The Go suite is derived from the Vitest cases
(about 1,900 tests and subtests) and reads the shared `tests/fixtures`.

Intentional differences from the TypeScript extension, all forced by PiG:

- Extension-only model fields live in the `litellm-model-policies.json` sidecar (see Design decisions), and the
  discovery-version staleness gate reads the sidecar instead of the stored model.
- The stored-catalog seed (`models_store.go`) reads the discovery version from the sidecar, and treats an undecodable
  stored record as the restore failure, because Go's `EnrichCachedModel` cannot fail where the TypeScript one throws.
  The typed store now rejects a malformed record on the host side before the extension sees it, so only
  extension-only fields (`litellmPolicy` and the like) can still fail there.
- PiG 0.5.2 emits `model_select` from `coding/session.go`, with `data["model"]`, `data["previousModel"]` and
  `data["source"]`. The budget status line follows the provider in `data["model"]["provider"]`;
  `before_agent_start`, `turn_end` and `session_start` remain as a fallback through `ctx.ModelProvider()` for a host
  that does not emit it.
- The first MCP sync runs at `session_start`, not while the factory runs: `*sdk.Extension.RegisterMcpServer` only
  queues the declaration and the host validates it while loading, so a refused config (`mcp.exposure: "bogus"`)
  would fail the whole extension load. `sdk.Context.RegisterMcpServer` returns the host's error, which is reported
  once through `notifyMcp`, and the registered server connects through `mcp_servers_change`.
  The access check itself starts while the factory runs, before seeding, and the first sync (the one at
  `session_start`) uses its answer. The early checks and the check's cleanup `DELETE` run detached through
  `catalog.Go`, and session shutdown drains them together with the catalog fetches. Shutdown first cancels the
  early checks no sync has used, so the next session asks again; the `DELETE` keeps its own timeout, because
  cancelling it would leave open the session the proxy opened for the check.
- Handlers and setup functions depend on two narrow unexported interfaces in `extension.go`, `hookRegistrar`
  (`OnEvent`, `Command`, `RegisterTool`; `sdkRegistrar` adapts `*sdk.Extension`) and `hookHost` (the `sdk.Context`
  methods the handlers use; `sdkHost` adapts `sdk.Context`), so the wiring tests register into a recording fake and
  call the handlers with hand-built `data` maps (`TestHookRegistration`, `TestHookDataKeysAndScoping`).
- `ctx.ui.setStatus(undefined)` has no SDK equivalent; an empty string clears the budget status.
- Diagnostics the TypeScript writes to stderr go through one reporter; in a fused binary they still reach stderr.
- Image generation is not ported. The TypeScript extension publishes `mode: "image_generation"` route groups as Pi
  image models served through `/v1/images/generations` (`src/images.ts`), but `internal/bridge` maps neither
  `GenerateImages` nor `GetAllModels`, so a PiG native provider cannot surface image models. The Go discovery keeps
  withholding those groups like any other non-chat mode; port `src/images.ts` once the SDK exposes both callbacks.
  Classification is not ported either: `src/classifiers.ts` publishes decision route groups (`mode: "evaluation"` or
  `"decisions"`, or a decision endpoint in `supported_endpoints`) as Pi classifier models served through
  `/v1/systemone`, the bridge maps neither `Classify` nor `GetAllModels`, and the Go discovery keeps withholding those
  groups like any other non-chat mode.

## Known PiG 0.5.2 issues

- A reasoning model whose `thinkingLevelMap` denies every level (the TypeScript's `NO_TRANSMISSIBLE_LEVELS`, emitted
  when no effort carrier is evidenced, for example a Claude route on Chat Completions without
  `reasoning_effort_levels`) crashes PiG at model selection: `coding/model.go:175` (`thinkingMaxLevelForEntry`) indexes
  the last element of the empty list `ai.GetSupportedThinkingLevels` returns. Pi tolerates the same model. The fix
  belongs in PiG (return `""` when the list is empty); the extension deliberately keeps the TypeScript semantics rather
  than inventing a level.
- PiG 0.5.x compiles a Go extension against the host's own SDK and ignores the `go.mod` SDK pin of the extension.
  SDK 0.5.2's `ModelRegistry.Find` also returns an error, so this extension needs PiG 0.5.2 or later; a 0.5.1 host
  cannot load it.
- When the extension fails to build, `--list-models` exits 0 and prints only `No models available`, while `-p` and
  `pig install --validate-only --json` report the build error.
- Fixed in 0.5.1: `/login litellm` offers the login flows under their `loginLabel`, answers `select` prompts and runs
  the provider's own API-key login ([MichaelKinsy/PiG#201](https://github.com/MichaelKinsy/PiG/issues/201)).
- Fixed in 0.5.1: fused builds accept third-party modules ([MichaelKinsy/PiG#196](https://github.com/MichaelKinsy/PiG/issues/196)),
  though this module still depends only on the standard library, PiG and the SDK by choice.
