# pig-provider-litellm

Go port of the [LiteLLM provider extension](../README.md) for [PiG](https://github.com/MichaelKinsy/PiG). It
registers the same `litellm` native provider (plus aliases from settings and from the `LITELLM_PROVIDERS`,
`LITELLM_PROVIDERS_JSON` and `LITELLM_PROVIDER_<NAME>_*` environment variables, as in the root README), discovers
models from your LiteLLM proxy, and adds LiteLLM MCP, Skill Hub, budget status and Google ADC support.

## Relation to the TypeScript extension

The TypeScript extension in `../src` is the specification; `extensions/litellm` ports its behavior (see
[PORTING.md](PORTING.md)). Both use the same configuration, so one setup works for Pi and PiG:

- the `litellm` settings block (`providers`, `mcp`, `budget`, `skills`),
- every `LITELLM_*` environment variable,
- the `auth.json` entries (`oauth` and `api_key`).

PiG reads them from its own agent directory (`~/.pig/agent`, or `PIG_CODING_AGENT_DIR`) instead of Pi's. See the
[root README](../README.md) for the settings reference.

## Install

```bash
pig install 'git:https://github.com/balcsida/pi-provider-litellm.git#subdirectory=pig'
```

Try it from a checkout without installing:

```bash
pig -e ./pig/extensions/litellm
```

### Fused binary

`litellm-example.yaml` is a Piglet that compiles the extension into a single PiG binary
(`build.extensionRealization: fused`):

```bash
pig piglet build pig/litellm-example.yaml --format binary --out ./pig-litellm
./pig-litellm --version
```

The first build downloads the PiG source and needs a Go toolchain (`pig setup go`).

## Configuration

Besides the `LITELLM_*` variables and the `litellm` block of `settings.json`, providers can be configured in
`models.json` in the PiG agent directory (`providers.litellm` for the default provider, `litellm.providers.<name>`
for aliases). It is read as JSONC like PiG's own reader (BOM, `//` comments, trailing commas) and wins per field
over `settings.json`; only providers are taken from it.

## Known PiG 0.5.1 issues

### Requires PiG 0.5.1 or later

PiG 0.5.x compiles a Go extension against the host's own SDK and ignores the SDK pin in the extension's `go.mod`, so
a 0.4.1 host cannot load this extension. When the extension fails to build, `--list-models` exits 0 and prints only
`No models available`; `-p` and `pig install --validate-only --json` report the build error. On stock 0.5.1,
`/login litellm` offers the SSO flows.

### Reasoning models without an effort carrier crash model selection

When discovery finds a reasoning model but no evidence of a transmissible effort level (for example a Claude route
served over Chat Completions without `reasoning_effort_levels`), it emits a thinking-level map that denies every
level, exactly as the TypeScript extension does. PiG 0.5.1 then panics while selecting that model
(`coding/model.go:180`, `thinkingMaxLevelForEntry` indexes the last element of an empty list), whether through
`--model` or `/model`. Pi tolerates the same model. Until PiG guards the empty list, pick a route that declares its
effort levels, or route the model through the Messages API.

## Development

```bash
cd pig/extensions/litellm
gofmt -l .            # must print nothing
go vet ./...
go test -race ./...

cd ../../tools/mockproxy && go test ./...
pig package validate ./pig --json   # from the repository root
```

`tools/mockproxy` is a fake LiteLLM proxy for end-to-end checks without network:

```bash
(cd pig/tools/mockproxy && go run . -key test-key -addr 127.0.0.1:48417) &
export PIG_CODING_AGENT_DIR=$(mktemp -d) LITELLM_BASE_URL=http://127.0.0.1:48417 LITELLM_API_KEY=test-key
pig -e ./pig/extensions/litellm --list-models
pig -e ./pig/extensions/litellm --model litellm/mock-chat -p "say hi"
```

Repeat the prompt with `mock-responses` and `mock-claude`; the mock logs `POST /v1/responses` and `POST /v1/messages`
when each model reaches its own protocol's endpoint.

The mock also plays the proxy's SSO endpoints (`-sso cli|pkce|paste`), and `pig/tools/login-smoke/login-smoke.mts`
drives `/login litellm` through a real terminal against them:

```bash
(cd pig/tools/mockproxy && go run . -key test-key -addr 127.0.0.1:48417 -sso cli) &
PIG_BIN=pig LITELLM_BASE_URL=http://127.0.0.1:48417 LOGIN_SMOKE_MODE=cli npx tsx pig/tools/login-smoke/login-smoke.mts
```

`LOGIN_SMOKE_MODE` is `cli`, `pkce`, `paste` or `apikey`. It needs a PiG with the login fixes described above.

The `PiG` GitHub workflow runs these checks on every push and pull request.
