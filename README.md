# pi-provider-litellm

LiteLLM proxy provider extension for [Pi](https://pi.dev).

Discovers models from a self-hosted LiteLLM proxy and registers them under the `litellm` provider. Supports `/login litellm` and `/litellm-refresh`. Tries `/model/info` first (admin endpoint with rich metadata), falls back to `/v1/models` (OpenAI-compatible) on 401/403/404.

## Install

```bash
pi install npm:pi-provider-litellm
```

Pi fetches the package from npm and registers it. Add `-l` to install into project settings (`.pi/settings.json`) instead of global.

After installing, you can pass the LiteLLM host when starting pi:

```bash
pi --litellm-host https://litellm.your-domain.com
```

To try it without installing (one-off, current run only):

```bash
pi -e npm:pi-provider-litellm
```

With a one-off extension run, pass the host after the extension source:

```bash
pi -e npm:pi-provider-litellm --litellm-host https://litellm.your-domain.com
```

<details>
<summary>Alternative: install from source</summary>

```bash
git clone https://github.com/balcsida/pi-provider-litellm.git ~/.pi/agent/extensions/pi-provider-litellm
cd ~/.pi/agent/extensions/pi-provider-litellm
npm ci
npm run clean && npm run build
```

</details>

## Configure

### Option A — interactive login

Inside pi:

```
/login litellm
```

You'll be prompted for the base URL and API key. Credentials are persisted to `~/.pi/agent/auth.json`.

### Option B — environment variables

```bash
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_API_KEY="sk-..."
```

Stored pi credentials for `litellm` take precedence over `LITELLM_API_KEY`; the environment key is used when no saved credential exists. `LITELLM_BASE_URL` is used when no saved login base URL exists.

### Option C — CLI host parameter

```bash
pi --litellm-host https://litellm.your-domain.com
```

`--litellm-host` supplies the LiteLLM base URL for the current pi run and accepts URLs with or without a trailing `/v1`. `--litellm-base-url` is also supported as an alias. The CLI host parameter takes precedence over saved login base URLs and `LITELLM_BASE_URL`; API keys still come from `/login litellm` or `LITELLM_API_KEY`.

## Use

```
/model
```

## Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `LITELLM_OFFLINE` | unset | If `1`, skip discovery on this start; use cache only |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Discovery fetch timeout in ms; `0` to skip discovery |

`LITELLM_DISCOVERY_TIMEOUT_MS=0` only disables startup and refresh model discovery. It does not replace the base URL or API key settings required to send requests when you are not using `/login litellm`.

## Mocked LiteLLM smoke workflow

The `LiteLLM Smoke` GitHub Actions workflow starts VidaiMock and a real LiteLLM proxy on the runner. LiteLLM exposes OpenAI-compatible and Anthropic routes whose upstreams are served by VidaiMock, then this extension's smoke runner discovers those models through LiteLLM and sends `/v1/chat/completions` requests through the proxy.

This keeps the LiteLLM integration path under test but does not call real LLM APIs. No provider API keys or GitHub Models permission are required. The workflow also runs a non-interactive Pi CLI smoke with `--list-models` and `-p` so extension loading, model discovery, and a real completion path are covered without opening the TUI.

## Development

This package requires Node.js `>=22.19.0`. CI currently uses Node `24.16.0`.

```bash
npm ci
npm run check
npm run clean && npm run build
```

`npm run check` runs Biome, type checking, and the Vitest suite. Runtime changes must be built before local Pi smoke checks because the extension entrypoint is `./dist/index.js`.

Before changing package contents or dependency policy, also run:

```bash
npm run supply-chain:guard
npm pack --dry-run
```

The published npm package should contain only `dist`, `README.md`, and `LICENSE`.

## Release

Releases are driven by semver tags named `v*.*.*`. The GitHub release workflow installs from the lockfile, runs the checks, builds `dist`, verifies the package tarball, publishes to npm with provenance, and creates a GitHub release.

Before tagging a release, keep `package.json` and `package-lock.json` versions in sync and verify the dry-run package contents.

## Slash commands

- `/litellm-refresh` — force re-fetch the model list, ignoring cache

## Cache

The model list is cached at `~/.pi/agent/litellm-models.json` with a keyed fingerprint of the base URL + API key. Changing either invalidates the cache automatically.

If the cache is older than 24 hours, the extension refreshes it in the background on session start (non-blocking). Failures are silent — the cached models remain in use. Run `/litellm-refresh` to force an immediate refresh.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| "discovered no models" | Proxy returned an empty list — check pi's startup log for the actual response |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models |

## License

MIT — see [LICENSE](./LICENSE).
