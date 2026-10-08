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
    ├── go.mod                      requires github.com/MichaelKinsy/PiG v0.4.1 and .../extensions/sdk v0.4.1
    ├── extension.go                factory: func Extension() *sdk.Extension      ← src/index.ts
    ├── provider.go                 LiteLLM native provider                       ← src/provider.ts
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
aliases come from settings as in the TypeScript version.

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

## Verification ladder

1. `go test ./...` in `pig/extensions/litellm`.
2. `pig install --validate-only --json ./pig/extensions/litellm` proves the factory builds, but PiG 0.4.1's
   validate host never binds the native provider registry, so any extension that registers a native provider
   fails it with `native provider registry is not bound`. Treat that exact error as expected there; a real load
   is the test.
3. Live smoke against the mock proxy in `pig/tools/mockproxy` (`go run . -key test-key -addr 127.0.0.1:48417`):

   ```sh
   export PIG_CODING_AGENT_DIR=$(mktemp -d) LITELLM_BASE_URL=http://127.0.0.1:48417 LITELLM_API_KEY=test-key
   pig -e ./pig/extensions/litellm --list-models            # lists mock-chat, mock-claude, mock-responses
   pig -e ./pig/extensions/litellm --model litellm/mock-chat -p "say hi"   # streams "mock reply to: ..."
   ```

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
   Stock PiG 0.4.1 fails this step (see Known PiG 0.4.1 issues); the fork branch named there passes all four.

## Status

All six phases are ported: discovery and model-group reduction, the native provider with the request guard and
refresh wrapper, credentials and provider auth, request and response policy hooks with cost tracking, the login
flows (CLI PKCE, direct OIDC, CLI SSO, pasted token) with refresh, budget status, Skills tools, MCP registration,
the PiG package manifest, an example fused Piglet and a CI workflow. The Go suite is derived from the Vitest cases
(about 1,900 tests and subtests) and reads the shared `tests/fixtures`.

Intentional differences from the TypeScript extension, all forced by PiG:

- Extension-only model fields live in the `litellm-model-policies.json` sidecar (see Design decisions), and the
  discovery-version staleness gate reads the sidecar instead of the stored model.
- PiG 0.4.1 emits `model_select` (from `SetModel`, with the Pi-shaped model in `data["model"]`), although its
  parity table still lists it as planned. The budget status line follows the provider in `data["model"]["provider"]`;
  `before_agent_start`, `turn_end` and `session_start` remain as a fallback through `ctx.ModelProvider()` for a host
  that does not emit it.
- The first MCP sync runs at `session_start`, not while the factory runs: `*sdk.Extension.RegisterMcpServer` only
  queues the declaration and the host validates it while loading, so a refused config (`mcp.exposure: "bogus"`)
  would fail the whole extension load. `sdk.Context.RegisterMcpServer` returns the host's error, which is reported
  once through `notifyMcp`, and the registered server connects through `mcp_servers_change`.
- Handlers and setup functions depend on two narrow unexported interfaces in `extension.go`, `hookRegistrar`
  (`OnEvent`, `Command`, `RegisterTool`; `sdkRegistrar` adapts `*sdk.Extension`) and `hookHost` (the `sdk.Context`
  methods the handlers use; `sdkHost` adapts `sdk.Context`), so the wiring tests register into a recording fake and
  call the handlers with hand-built `data` maps (`TestHookRegistration`, `TestHookDataKeysAndScoping`).
- `ctx.ui.setStatus(undefined)` has no SDK equivalent; an empty string clears the budget status.
- A JSON `null` under `litellm.oidc` is treated as unset rather than rejected, because the decoded settings map
  cannot distinguish null from absent.
- Diagnostics the TypeScript writes to stderr go through one reporter; in a fused binary they still reach stderr.

## Known PiG 0.4.1 issues

- `pig install --validate-only` cannot validate an extension that registers a native provider (see step 2 above).
  `pig login --list` and the CLI `pig login litellm` start the extension through the same inspection host and fail
  the same way (`extension "litellm" inspection failed: ... native provider registry is not bound`); a real
  session binds the registry, so configure credentials through the environment or `settings.json`, or log in
  inside a session.
- `/login litellm` inside a stock 0.4.1 session offers only PiG's generic `Enter API key` prompt, stores the key
  without the proxy URL, and never shows the SSO flows. The extension declares both methods correctly and the host
  publishes the OAuth flow; the gaps are PiG's, fixed on the fork branch `fix/native-provider-login` of
  `balcsida/PiG`: the registry records a native provider's registration without its OAuth method and rewrites it on
  every catalog refresh (0.4.1 also predates PiG#192, the inheritance of a published flow); a flow's `select` prompt
  is answered with `Login cancelled`; a provider's own `apiKey.login` is not run; and the flow's `loginLabel` is not
  shown. With that build, step 5 of the verification ladder passes for CLI SSO, PKCE (with refresh), pasted token
  and API key, with no change to the extension. Until PiG ships it, configure credentials through the environment or
  `settings.json`.
- `pig piglet build` cannot fuse a Go factory that depends on a third-party module
  ([MichaelKinsy/PiG#196](https://github.com/MichaelKinsy/PiG/issues/196)): the fused builder overlays Pig's own
  `go.mod` and compiles read-only, and a Binary fuses every compatible factory whether or not
  `extensionRealization` is set. This is why the login flows stay hand-written instead of using
  `github.com/balcsida/litellm-auth-go`, and why this module must keep depending only on the standard library,
  PiG and the SDK until that issue is resolved.
- A reasoning model whose `thinkingLevelMap` denies every level (the TypeScript's `NO_TRANSMISSIBLE_LEVELS`, emitted
  when no effort carrier is evidenced, for example a Claude route on Chat Completions without
  `reasoning_effort_levels`) crashes PiG at model selection: `coding/model.go:173` (`thinkingMaxLevelForEntry`) indexes
  the last element of the empty list `ai.GetSupportedThinkingLevels` returns. Pi tolerates the same model. The fix
  belongs in PiG (return `""` when the list is empty); the extension deliberately keeps the TypeScript semantics rather
  than inventing a level.
