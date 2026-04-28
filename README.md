# @balcsida/pi-provider-litellm

[![npm version](https://img.shields.io/npm/v/@balcsida/pi-provider-litellm.svg)](https://www.npmjs.com/package/@balcsida/pi-provider-litellm)

LiteLLM proxy provider extension for [pi-mono](https://github.com/badlogic/pi-mono).

Discovers models from a self-hosted LiteLLM proxy and registers them under the `litellm` provider. Supports `/login litellm` and `/litellm-refresh`. Tries `/model/info` first (admin endpoint with rich metadata), falls back to `/v1/models` (OpenAI-compatible) on 401/403/404.

## Install

```bash
pi install npm:@balcsida/pi-provider-litellm
```

Pi fetches the package from npm and registers it. Add `-l` to install into project settings (`.pi/settings.json`) instead of global.

To try it without installing (one-off, current run only):

```bash
pi -e npm:@balcsida/pi-provider-litellm
```

<details>
<summary>Alternative: install from source</summary>

```bash
git clone https://github.com/balcsida/pi-provider-litellm.git ~/.pi/agent/extensions/pi-provider-litellm
cd ~/.pi/agent/extensions/pi-provider-litellm
npm install
npm run build
```

</details>

## Configure

### Option A — environment variables

```bash
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_API_KEY="sk-..."
```

### Option B — interactive login

Inside pi:

```
/login litellm
```

You'll be prompted for the base URL and API key. Credentials are persisted to `~/.pi/agent/auth.json`.

## Use

```
/model litellm/anthropic/claude-3-5-sonnet
```

(Whatever model IDs your proxy exposes — see the discovered list at startup.)

## Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `LITELLM_OFFLINE` | unset | If `1`, skip discovery on this start; use cache only |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Discovery fetch timeout in ms; `0` to skip discovery |

## Slash commands

- `/litellm-refresh` — force re-fetch the model list, ignoring cache

## Cache

The model list is cached at `~/.pi/agent/litellm-models.json` with a sha256 fingerprint of the base URL + API key. Changing either invalidates the cache automatically.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| "discovered no models" | Proxy returned an empty list — check pi's startup log for the actual response |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models |

## License

MIT — see [LICENSE](./LICENSE).
