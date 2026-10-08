# pig-provider-litellm

Go port of the [LiteLLM provider extension](../README.md) for [PiG](https://github.com/MichaelKinsy/PiG). It
registers the same `litellm` native provider (plus aliases from settings), discovers models from your LiteLLM
proxy, and adds LiteLLM MCP, Skill Hub, budget status and Google ADC support.

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

## Known PiG 0.4.1 issues

### Fused builds and third-party Go modules

`pig piglet build` cannot fuse a Go factory that depends on a third-party module
([MichaelKinsy/PiG#196](https://github.com/MichaelKinsy/PiG/issues/196)). This extension therefore depends only on
the standard library, PiG and the SDK, and its login flows are implemented here rather than through a shared
library.

### `--validate-only`

`pig install --validate-only` fails with `native provider registry is not bound` for this extension. PiG 0.4.1's
validate host does not bind the native provider registry, and this extension registers one. Use
`pig package validate ./pig` to validate the package, and a real load (`pig -e ...`) to prove the extension works.
The same inspection host backs `pig login --list` and the CLI `pig login litellm`, which report
`extension "litellm" inspection failed` on 0.4.1; configure credentials through the environment or
`settings.json`, or log in inside a session, where the registry is bound.

### Reasoning models without an effort carrier crash model selection

When discovery finds a reasoning model but no evidence of a transmissible effort level (for example a Claude route
served over Chat Completions without `reasoning_effort_levels`), it emits a thinking-level map that denies every
level, exactly as the TypeScript extension does. PiG 0.4.1 then panics while selecting that model
(`coding/model.go:173`, `thinkingMaxLevelForEntry` indexes the last element of an empty list), whether through
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

The `PiG` GitHub workflow runs these checks on every push and pull request.
