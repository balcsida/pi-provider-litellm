# pi-provider-litellm

LiteLLM proxy provider extension for [Pi](https://pi.dev).

Discovers models from a self-hosted LiteLLM proxy and registers them under the `litellm` provider. Supports `/login litellm` and `/litellm-refresh`. Tries `/model/info` first (admin endpoint with rich metadata), falls back to `/v1/models` (OpenAI-compatible) on 401/403/404.

## Install

```bash
pi install npm:pi-provider-litellm
```

Pi fetches the package from npm and registers it. Add `-l` to install into project settings (`.pi/settings.json`) instead of global.

To try it without installing (one-off, current run only):

```bash
pi -e npm:pi-provider-litellm
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

## Real LiteLLM smoke workflow

The `LiteLLM Smoke` GitHub Actions workflow starts a real LiteLLM proxy on the runner, discovers models through this extension's smoke runner, and sends real `/v1/chat/completions` requests through the proxy.

The required baseline route uses GitHub Models with the workflow `GITHUB_TOKEN` and `models: read` permission, so it can run without adding provider secrets. Direct vendor routes are added only when these repository secrets exist:

| Secret | Default smoke model | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `gpt-4.1-nano` | Cheap direct OpenAI/GPT route |
| `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | Cheap direct Claude route |
| `GEMINI_API_KEY` | `gemini-2.5-flash-lite` | Cheap/free-tier-friendly Gemini route |

Optional repository variables override model choices without editing the workflow:

| Variable | Default |
|---|---|
| `GH_MODELS_SMOKE_MODEL` | `openai/gpt-4o-mini` |
| `OPENAI_SMOKE_MODEL` | `gpt-4.1-nano` |
| `ANTHROPIC_SMOKE_MODEL` | `claude-haiku-4-5-20251001` |
| `GEMINI_SMOKE_MODEL` | `gemini-2.5-flash-lite` |

Manual runs include a `require_vendors` input. When enabled, the workflow fails before starting LiteLLM unless all three direct vendor secrets are configured. Scheduled and push runs keep the GitHub Models baseline required and skip missing direct vendors.

## Slash commands

- `/litellm-refresh` — force re-fetch the model list, ignoring cache

## Cache

The model list is cached at `~/.pi/agent/litellm-models.json` with a keyed fingerprint of the base URL + API key. Changing either invalidates the cache automatically.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| "discovered no models" | Proxy returned an empty list — check pi's startup log for the actual response |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models |

## License

MIT — see [LICENSE](./LICENSE).
