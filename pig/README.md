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
([MichaelKinsy/PiG#196](https://github.com/MichaelKinsy/PiG/issues/196); a fix is offered in
[MichaelKinsy/PiG#202](https://github.com/MichaelKinsy/PiG/pull/202)). This extension therefore depends only on
the standard library, PiG and the SDK, and its login flows are implemented here rather than through a shared
library.

### `--validate-only`

`pig install --validate-only` fails with `native provider registry is not bound` for this extension. PiG 0.4.1's
validate host does not bind the native provider registry, and this extension registers one. Use
`pig package validate ./pig` to validate the package, and a real load (`pig -e ...`) to prove the extension works.
The same inspection host backs `pig login --list` and the CLI `pig login litellm`, which report
`extension "litellm" inspection failed` on 0.4.1; configure credentials through the environment or
`settings.json`, or log in inside a session, where the registry is bound.

### Logging in inside a session

On stock PiG 0.4.1, `/login litellm` offers only PiG's generic API-key prompt: the SSO flows (CLI SSO, PKCE,
direct OIDC, pasted token) never appear, and the key is stored without the proxy URL. PiG records a native
provider's registration without its account login method, answers a flow's `select` prompt with `Login cancelled`,
and does not run a provider's own API-key login ([MichaelKinsy/PiG#201](https://github.com/MichaelKinsy/PiG/issues/201)).
The fork branch `fix/native-provider-login` of `balcsida/PiG` fixes all of it without any change here; with that build the login smoke in `pig/tools/login-smoke` passes for CLI
SSO, PKCE with refresh, pasted token and API key. Until PiG ships the fix, configure credentials through
`LITELLM_API_KEY` and `LITELLM_BASE_URL` or `settings.json`.

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

The mock also plays the proxy's SSO endpoints (`-sso cli|pkce|paste`), and `pig/tools/login-smoke/login-smoke.mts`
drives `/login litellm` through a real terminal against them:

```bash
(cd pig/tools/mockproxy && go run . -key test-key -addr 127.0.0.1:48417 -sso cli) &
PIG_BIN=pig LITELLM_BASE_URL=http://127.0.0.1:48417 LOGIN_SMOKE_MODE=cli npx tsx pig/tools/login-smoke/login-smoke.mts
```

`LOGIN_SMOKE_MODE` is `cli`, `pkce`, `paste` or `apikey`. It needs a PiG with the login fixes described above.

The `PiG` GitHub workflow runs these checks on every push and pull request.
