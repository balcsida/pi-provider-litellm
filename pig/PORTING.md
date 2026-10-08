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
2. `pig install --validate-only --json ./pig/extensions/litellm` builds and registers the factory.
3. `pig -e ./pig/extensions/litellm` against a local mock proxy (phase 2 adds the mock).
4. `pig piglet build` of a sample Piglet with `extensionRealization: fused` (phase 6).
