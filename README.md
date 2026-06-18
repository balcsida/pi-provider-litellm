# pi-provider-litellm

LiteLLM proxy provider extension for [Pi](https://pi.dev).

Discovers models from a self-hosted LiteLLM proxy and registers them under the `litellm` provider. Supports `/login litellm`, `/litellm-refresh`, LiteLLM MCP tools, LiteLLM Skills Gateway prompt injection, and Google ADC token auth. Tries `/model/info` first (admin endpoint with rich metadata), falls back to `/v1/models` (OpenAI-compatible) on 401/403/404, then tries `/health` plus per-endpoint `/model/info` for older LiteLLM proxies.

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

You can also run `/login`, select `Use a subscription`, then select `LiteLLM`.

You'll be prompted for the base URL and API key. Credentials are persisted to `~/.pi/agent/auth.json`.

#### Enterprise SSO login

If your LiteLLM proxy requires SSO/OAuth authentication (enterprise deployments), you can authenticate via a browser SSO flow and optionally pair the resulting JWT with a stable virtual key:

1. Run `/login litellm` inside pi
2. Enter the proxy URL
3. At the login method prompt, enter `2` for SSO / Enterprise JWT
4. Your default browser opens the LiteLLM SSO login URL (e.g. `https://litellm.your-domain.com/sso/key/generate`) automatically — the URL is also displayed in case it can't be opened. Authenticate via SSO
5. Copy your token from the LiteLLM UI and paste it at the prompt (copying a full `Bearer ...` header value is fine — the prefix is stripped automatically)
6. When prompted to generate a virtual key, press Enter to accept (recommended) or enter `n` to use the JWT directly

When you generate a virtual key, the resulting `sk-...` key is stored as your credential and used for all API requests. If the proxy's key policy attaches an expiry to the generated key, Pi will prompt you to re-authenticate when it nears expiry; otherwise the key is treated as permanent until revoked in LiteLLM.

When using a JWT directly, the extension reads its `exp` claim and Pi will prompt you to re-authenticate when the token nears expiry. Run `/login litellm` again to refresh.

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
| `LITELLM_API_KEY_HELPER` | unset | Command that prints a fresh LiteLLM bearer token. Takes precedence over `LITELLM_API_KEY`. Registered as a `!command` provider key; Pi re-runs it on every request (the per-request auth path is uncached), so rotating/short-lived tokens stay fresh. |
| `LITELLM_GCLOUD_TOKEN_AUTH` | unset | If set to a non-empty value other than `0`, use Google Application Default Credentials as the LiteLLM bearer token source when no stored `/login litellm` credential exists. If ADC is unavailable, the extension falls back to `LITELLM_API_KEY_HELPER` or `LITELLM_API_KEY`. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google default ADC path | Optional path to an ADC JSON file used by `LITELLM_GCLOUD_TOKEN_AUTH`. If unset, the extension checks the default gcloud ADC locations. |
| `LITELLM_OFFLINE` | unset | If `1`, skip discovery on this start; use cache only |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Discovery fetch timeout in ms; `0` to skip discovery |

`LITELLM_DISCOVERY_TIMEOUT_MS=0` only disables startup and refresh model discovery. It does not replace the base URL or API key settings required to send requests when you are not using `/login litellm`.

### Google ADC token auth

When your LiteLLM proxy accepts Google OAuth access tokens, you can let the extension refresh tokens from Application Default Credentials:

```bash
gcloud auth application-default login
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_GCLOUD_TOKEN_AUTH=1
```

Only `authorized_user` ADC files are supported. Service account JSON files are rejected with a warning. Tokens are cached in memory for 50 minutes and are keyed by the ADC identity, so switching `GOOGLE_APPLICATION_CREDENTIALS` or changing `gcloud auth application-default login` users refreshes both request auth and model discovery cache identity. When ADC is available, the registered provider key is a Pi `!command`, so request-time auth resolves a fresh token when Pi sends model requests. If ADC cannot be read, provider registration uses the same helper or environment API key that discovery used.

## LiteLLM MCP tools

If your LiteLLM proxy exposes MCP REST endpoints, this extension discovers tools from:

- `GET /mcp-rest/tools/list`
- `POST /mcp-rest/tools/call`

Each discovered tool is registered as a native Pi tool named `mcp_<server>_<tool>`, with simple JSON Schema parameters mapped to Pi/TypeBox parameters. Complex schemas fall back to a single `args` object. The extension accepts LiteLLM's `tools` response shape with `inputSchema` and `mcp_info`, and executes tools with LiteLLM's `server_id`, `name`, and `arguments` fields. MCP discovery runs after successful live model discovery, `/login litellm`, or `/litellm-refresh`; it does not force network or helper-token access when a fresh model cache is used.

## LiteLLM Skills Gateway

If your LiteLLM proxy exposes `/v1/skills`, enabled skills are fetched before each agent turn and appended to the system prompt as a `litellm_skills` section. The extension also registers Pi tools for basic Skills Gateway management:

- `litellm_skill_list`
- `litellm_skill_create`
- `litellm_skill_delete`

## Slash commands

- `/litellm-refresh` — force re-fetch the model list, ignoring cache

## Cache

The model list is cached at `~/.pi/agent/litellm-models.json` with the base URL and a keyed fingerprint of the selected auth source. Changing the base URL, API key, helper command, or Google ADC identity invalidates the cache automatically.

If the cache is older than 24 hours, the extension refreshes it in the background on session start (non-blocking). Failures are silent — the cached models remain in use. Run `/litellm-refresh` to force an immediate refresh.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| "discovered no models" | Proxy returned an empty list — check pi's startup log and verify `/model/info`, `/v1/models`, or `/health` responds |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models |
| `401 Token expired` | Set `LITELLM_API_KEY_HELPER`. |
| No models with gcloud auth | Verify `gcloud auth application-default login` has been run or set `GOOGLE_APPLICATION_CREDENTIALS` to an `authorized_user` ADC file |
| Enterprise SSO login shows "virtual key generation failed" | The LiteLLM instance may lack a database (`/key/generate` requires one), your user account may lack key-generation permission, or the request timed out; the JWT is used directly as a fallback |
| Enterprise SSO token prompt fails with "SSO token is required" | The token field was left empty — paste the token copied from the LiteLLM UI |
| MCP tools not showing | Verify the proxy exposes `/mcp-rest/tools/list` and run `/litellm-refresh` after fixing the proxy |
| Skills not affecting prompts | Verify the proxy exposes `/v1/skills` and returns enabled skills |

## License

MIT — see [LICENSE](./LICENSE).
