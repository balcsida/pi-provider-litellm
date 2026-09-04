# pi-provider-litellm

LiteLLM proxy native Provider extension for [Pi](https://pi.dev). Pi 0.83.0+ is required; the TypeScript source
entrypoint has been smoke-tested with Pi 0.81.0.

Discovers models from self-hosted LiteLLM proxies and registers them under Pi providers. The default provider is `litellm`; optional aliases can register additional LiteLLM providers with separate credentials. Supports `/login litellm`, LiteLLM MCP tools, LiteLLM Skills Gateway prompt injection, and Google ADC token auth. Tries `/model/info` first (admin endpoint with rich metadata), falls back to `/v1/models` (OpenAI-compatible) on 401/403/404, then tries `/health` plus per-endpoint `/model/info` for older LiteLLM proxies.

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
```

Pi loads the TypeScript source entrypoint declared in `package.json` `pi.extensions`, so a source install needs no
build step and no `node_modules`. Install dependencies only to run the test suite or the local checks.

The previously published 2.2.0 package was also importable as `pi-provider-litellm` from JavaScript or TypeScript. The source-only package no longer exposes that library import; it is supported only as a Pi extension.

</details>

## Configure

### Option A — interactive login

Inside pi:

```
/login litellm
```

To configure an API key, run `/login`, choose `Sign in with an API key`, then choose `LiteLLM API key`. With `/login litellm`, choose `Sign in with an API key` directly.

You'll be prompted for the base URL and API key. Credentials are persisted to `~/.pi/agent/auth.json`.

If a proxy URL is already known — from `providers.<name>.baseUrl`, `LITELLM_BASE_URL`, or a previous login — pi offers it as the first option instead of asking you to retype it. Pick `Enter a different URL…` to point at another proxy.

#### Enterprise SSO login

If your LiteLLM proxy requires SSO/OAuth authentication (enterprise deployments), you can authenticate through LiteLLM's CLI SSO flow:

1. Run `/login litellm` inside pi and select `Sign in with LiteLLM SSO`
2. Confirm the offered proxy URL, or enter one if this is the first login
3. Pi displays a verification code and opens the LiteLLM SSO login URL automatically
4. Authenticate in the browser and confirm the displayed code
5. If your account belongs to multiple teams, select the team for this session in Pi

Pi polls the proxy until authentication completes and stores the returned short-lived CLI token. Newer proxies advertise the token lifetime; for older proxies, Pi uses `LITELLM_CLI_JWT_EXPIRATION_HOURS` or LiteLLM's 24-hour default. On LiteLLM versions that gate CLI SSO, start the proxy with `EXPERIMENTAL_UI_LOGIN=True`.

Older proxies without `/sso/cli/start` fall back to the legacy browser flow: copy the token from the LiteLLM UI, paste it into Pi, and optionally exchange it for a virtual key. Pi reads JWT expiry claims and prompts you to run `/login litellm` again when re-authentication is required.

### Option B — environment variables

```bash
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_API_KEY="sk-..."
```

Stored pi credentials for `litellm` take precedence over `LITELLM_API_KEY`; the environment key is used when no saved credential exists. `LITELLM_BASE_URL` is used when no saved login base URL exists. Chat Completions and Responses models use the proxy root plus `/v1`; native Messages uses the proxy root directly.

### Multiple LiteLLM provider aliases

Add alias providers in `~/.pi/agent/settings.json` under `litellm.providers`. Each alias is registered as a separate Pi provider name, so models appear as `litellm/model-id` and `litellm-anthropic/model-id`.

```json
{
  "litellm": {
    "providers": {
      "litellm-anthropic": {
        "baseUrl": "https://litellm.your-domain.com",
        "apiKey": "$LITELLM_CLAUDE_KEY",
        "headers": "$LITELLM_HEADERS"
      }
    }
  }
}
```

You can also override the default provider through the same shape:

```json
{
  "litellm": {
    "providers": {
      "litellm": {
        "baseUrl": "https://litellm.your-domain.com",
        "apiKey": "$LITELLM_API_KEY",
        "headers": "$LITELLM_HEADERS"
      },
      "litellm-anthropic": {
        "baseUrl": "https://litellm.your-domain.com",
        "apiKey": "$LITELLM_CLAUDE_KEY",
        "headers": "$LITELLM_HEADERS"
      }
    }
  }
}
```

Provider fields:

| Field | Default | Effect |
|---|---|---|
| `baseUrl` | `LITELLM_BASE_URL` for `litellm`; required for aliases | LiteLLM proxy URL, with or without `/v1` |
| `apiKey` | `LITELLM_API_KEY_HELPER`/`LITELLM_API_KEY` for `litellm`; required for aliases | Pi config value for this provider's key. Use `$ENV_VAR`, `${ENV_VAR}`, `!command`, or a literal key. Escape a literal `$` as `$$`. |
| `headers` | `$LITELLM_HEADERS` for `litellm`; unset for aliases | JSON string env reference or inline object of request headers |
| `displayName` | provider name | Label shown in Pi UI |
| `enabled` | `true` | Set `false` to skip an alias |
| `allowInsecureHttp` | `false` | Set `true` to permit plaintext HTTP for this provider, for example `http://host.docker.internal`. Credentials and request data will not be encrypted. Loopback HTTP works without this setting. |

`/login litellm` and Google ADC token auth remain scoped to the default `litellm` provider. Aliases use their configured `apiKey` or manually stored auth entries matching the alias name.

### Optional LiteLLM features

LiteLLM Skills and MCP integration are enabled by default. Disable either feature globally in `~/.pi/agent/settings.json`:

```json
{
  "litellm": {
    "skills": {
      "enabled": false
    },
    "mcp": {
      "enabled": false
    }
  }
}
```

Setting `skills.enabled` to `false` disables the Skills Gateway management tools, skill fetching, and system-prompt injection. Setting `mcp.enabled` to `false` disables LiteLLM MCP discovery and tool registration. Restart Pi after changing these settings so previously registered tools are removed.

Treat the configured LiteLLM proxy as trusted: Skills can add instructions to the system prompt, and MCP can expose tools the agent may call. Disable these integrations when the proxy, its administrators, or its configured content are not fully trusted.

## Use

```
/model
```

## Model transport

A `/model/info` route group uses native Anthropic `/v1/messages` only when every deployment explicitly reports Chat mode, identifies a Claude backend through an Anthropic-, Bedrock-, or Vertex-capable adapter, and every declared backend identity resolves the same Anthropic compatibility policy. A `claude-<name>-<major>-<minor>` backend id resolves through its `claude-<name>-<major>` entry when the exact id is absent; an unknown generation stays on Chat Completions. When `supported_endpoints` is present, it must include `/v1/messages`; an explicit endpoint list without it keeps the group on Chat Completions. An unresolved routing or base-model identity denies native Messages rather than being discarded from the unanimity check. Mixed-generation, mixed-family, and unknown groups remain on Chat Completions; unanimous explicit Responses mode takes precedence. Routes discovered through the `/health` fallback stay on Chat Completions with reasoning controls closed, even when per-deployment `/model/info` detail agrees. Catalog identity such as Amazon Bedrock remains separate from the selected wire protocol.

Native Messages authenticates with `x-api-key`; body session metadata is sent only on the OpenAI-compatible transports.

## Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `LITELLM_API_KEY_HELPER` | unset | Command that prints a fresh LiteLLM bearer token. Takes precedence over `LITELLM_API_KEY`. The extension runs it while resolving request auth, and Pi's per-request auth path is uncached, so rotating/short-lived tokens stay fresh. |
| `LITELLM_HEADERS` | unset | JSON object of extra headers sent to LiteLLM provider, discovery, MCP, and Skills Gateway requests. Provider aliases can use it with `"headers": "$LITELLM_HEADERS"`. |
| `LITELLM_GCLOUD_TOKEN_AUTH` | unset | If set to a non-empty value other than `0`, use Google Application Default Credentials as the LiteLLM bearer token source. This takes precedence over `LITELLM_API_KEY_HELPER` and `LITELLM_API_KEY` when no stored `/login litellm` credential exists. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google default ADC path | Optional path to an ADC JSON file used by `LITELLM_GCLOUD_TOKEN_AUTH`. If unset, the extension checks the default gcloud ADC locations. |
| `LITELLM_OFFLINE` | unset | If `1`, disable all model and MCP discovery, including post-login discovery; use cached models only when their stored canonical proxy root exactly matches the active credential root, including any path prefix. URL-standard host casing and default ports are canonicalized, but paths remain case-sensitive. |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Background and explicit discovery fetch timeout in ms; `0` disables automatic discovery |
| `LITELLM_CLI_JWT_EXPIRATION_HOURS` | `24` | CLI SSO token lifetime fallback for older proxies whose poll response omits `expires_in`; mirror a non-default proxy setting locally |
| `LITELLM_VERBOSE_DISCOVERY` | unset | If `1`, enable progress messages during model and MCP discovery (login, refresh, startup); discovery is silent by default |

Only use a trusted `LITELLM_API_KEY_HELPER` or `!command`. Prefer an absolute executable path, keep secrets out of command arguments, and print only the token to stdout without logging it to stderr.

`LITELLM_DISCOVERY_TIMEOUT_MS=0` disables automatic and explicit refresh model discovery. It does not replace the base URL or API key settings required to send requests when you are not using `/login litellm`.

### Google ADC token auth

When your LiteLLM proxy accepts Google OAuth access tokens, you can let the extension refresh tokens from Application Default Credentials:

```bash
gcloud auth application-default login
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_GCLOUD_TOKEN_AUTH=1
```

Only `authorized_user` ADC files are supported. Service account JSON files are rejected with a warning. An `authorized_user` file with a missing, empty, or whitespace-only `client_id`, `client_secret`, or `refresh_token` is rejected before any token exchange. Tokens are refreshed in-process when Pi resolves request auth and cached in memory for 50 minutes.

## LiteLLM MCP tools

If your LiteLLM proxy exposes MCP REST endpoints, this extension discovers tools from:

- `GET /mcp-rest/tools/list`
- `POST /mcp-rest/tools/call`

Each discovered tool is registered as a native Pi tool named `mcp_<server>_<tool>`, with simple JSON Schema parameters mapped to Pi/TypeBox parameters. Complex schemas fall back to a single `args` object. MCP discovery runs after Pi refreshes LiteLLM models or after `/login litellm`; extension activation never waits for it. MCP tools run in Pi's parallel tool mode and retry transient failures once.

## LiteLLM Skill Hub

If your LiteLLM proxy exposes `/claude-code/marketplace.json`, enabled skills are fetched before each agent turn and appended to the system prompt as a `litellm_skills` section. The extension falls back to the legacy `/v1/skills` Skills Gateway path when Skill Hub is unavailable. It also registers Pi tools for basic skill management:

- `litellm_skill_list`
- `litellm_skill_create`
- `litellm_skill_delete`

## Mocked LiteLLM smoke workflow

The `LiteLLM Smoke` GitHub Actions workflow starts VidaiMock and a real LiteLLM proxy on the runner. LiteLLM exposes route-distinct Chat, Responses, native Messages, and mixed-deployment models whose upstreams are served by VidaiMock. The smoke runner discovers those models through LiteLLM, asserts each model's expected API, exercises `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`, verifies the expected `x-litellm-response-cost` behavior, and proves endpoint coverage from captured LiteLLM request logs rather than response text.

This keeps the LiteLLM integration path under test but does not call real LLM APIs. No provider API keys or GitHub Models permission are required. The smoke runner also asserts that discovery came from `/model/info` (`LITELLM_SMOKE_EXPECT_SOURCE`) so a silent fallback to `/v1/models` fails the run. The workflow also runs auth checks plus optional Postgres-backed auth checks when `LITELLM_LICENSE` is configured for virtual-key and admin-route behavior, then runs a non-interactive Pi CLI smoke with `--list-models` and `-p` against both the OpenAI-compatible and Anthropic-backed routes, so extension loading, model discovery, and real completion paths are covered without opening the TUI. It also runs an interactive Pi TUI smoke covering `/login litellm` and Pi's native `/model` refresh. VidaiMock returns fixed responses, so the real-proxy smoke does not exercise a model-originated tool call or thinking block; the native Messages compatibility suite covers thinking, tool use, tool results, and replay across turns.

## Development

This package requires Node.js `>=22.19.0`. CI currently uses Node `26.5.0`.

```bash
npm ci
npm run check
npm run clean && npm run build
```

`npm run check` runs Biome, type checking, the Vitest suite, and the supply-chain package-content guard. Pi installs and local smoke checks load the shipped `src/index.ts` entrypoint directly; `dist/` is verification output only.

Before changing package contents or dependency policy, also run:

```bash
npm run supply-chain:guard
npm pack --dry-run
```

The published npm package contains only `src`, `README.md`, `LICENSE`, and the `package.json` npm always includes. Pi loads the TypeScript source entrypoint for both npm and Git installs. The package does not expose a JavaScript or TypeScript library import; load it through Pi.

## Release

Releases are driven by semver tags named `v*.*.*`. The GitHub release workflow installs from the lockfile, then `npm publish` runs `prepublishOnly` to check the source, build `dist` for verification, verify the package contents, and publish to npm with provenance before the workflow creates a GitHub release.

Before tagging a release, keep `package.json` and `package-lock.json` versions in sync and verify the dry-run package contents.

## Model catalog

Dynamic catalogs are persisted by Pi in `~/.pi/agent/models-store.json`. Credentials remain in `~/.pi/agent/auth.json`. Legacy `litellm-models*.json` files are ignored and are not deleted.

Opening `/model` refreshes configured provider catalogs in the background using Pi's native model lifecycle.

### Model host enforcement

Models this provider dispatches are sent to the proxy root resolved from the active credential. The extension hides cached models whose stored canonical root is malformed, still uses the documentation placeholder, or differs from that credential, and its dispatch guard rejects the same cases before a request. Accepted request URLs are re-derived from the credential root rather than trusted from model configuration. Path prefixes are part of the root and remain case-sensitive.

Catalog filtering and dispatch are separate. Choosing a model by ID or restoring it from a session can skip filtering. Pi 0.84 routes such a model to this provider only when its current catalog already contains the same `api`; the native `Provider` contract has no separate protocol-capability declaration. If the catalog has no model for that API, Pi uses its global implementation, bypassing this extension's host guard while still applying the LiteLLM credential. Until Pi exposes provider protocol capabilities, do not configure or restore a LiteLLM model whose `api` is absent from the current catalog, and never give such an entry a `baseUrl` that should not receive the LiteLLM credential.

`LITELLM_OFFLINE=1` does not recover a root mismatch: refresh online against the active proxy first, then return to offline use.

### Deployment groups and metadata authority

LiteLLM may load-balance one public `model_name` across deployments with different backends or model versions. The extension reduces `/model/info` rows conservatively before publishing one Pi model:

- Responses is selected only when every row explicitly reports Responses mode; mixed or unknown groups use Chat.
- Vision and reasoning are advertised only when every routable deployment resolves them as supported. Router-reported reasoning-effort levels are exposed only when every deployment explicitly supports the level.
- Context and output limits use the minimum resolved value across deployments.
- Each displayed price field uses the maximum only when every deployment resolves that field; unresolved fields remain zero. A model name is suffixed with ` (incomplete metadata)` whenever any published capability, limit, or price remains unresolved.
- Catalog metadata is accepted only from one unanimous concrete model identity derived from deployment fields such as `litellm_params.model`, `model_info.base_model`, and the LiteLLM adapter. Different concrete backends conflict even when they belong to the same provider or semantic family. Ambiguous groups are not matched across all Pi provider catalogs. For example, a route spanning Anthropic and Bedrock Claude deployments keeps explicit router metadata but does not borrow either provider's catalog metadata.

The ` (no metadata)` suffix is reserved for evidence-free `/v1/models` fallback entries eligible for bounded catalog enrichment on a later cache read. A recognized `owned_by` provider that conflicts with a provider-qualified model ID instead receives ` (incomplete metadata)`, permanently withholding cache enrichment rather than falling through to the ID prefix. The ` (incomplete metadata)` suffix marks reduced `/model/info` groups or unresolved `/health` routes and permanently prevents route-name cache enrichment. It means at least one metadata field is unknown, including cache pricing that the proxy omitted; known input/output prices may still be shown alongside the suffix.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| `No models available` at startup, gone after a restart | A Pi startup race, not discovery — see [`No models available` at startup](#no-models-available-at-startup) |
| "discovered no models" | Proxy returned an empty list — check pi's startup log and verify `/model/info`, `/v1/models`, or `/health` responds |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models. Offline mode does not recover a root mismatch — see [Model host enforcement](#model-host-enforcement) |
| A provider shows no models | Its active root is missing, malformed, still the placeholder, or differs from the cached catalog root. Check stderr and see [Model host enforcement](#model-host-enforcement) |
| A configured or restored model bypasses host enforcement | Its `api` is absent from the current provider catalog, so Pi used global API fallback. Refresh the catalog and do not configure an untrusted `baseUrl` for that entry — see [Model host enforcement](#model-host-enforcement) |
| `LiteLLM discovery: ... route group(s) have missing or conflicting deployment provider evidence` | One or more deployments lack a resolvable backend provider or resolve to different providers. Add consistent `litellm_params.model`, `model_info.base_model`, or adapter metadata; catalog-derived limits, pricing, and reasoning metadata are withheld meanwhile. |
| A model is marked ` (incomplete metadata)` | `/model/info` or `/health` did not provide enough authoritative metadata. Explicit fields remain usable, but unknown cost fields are shown as zero and route-name cache enrichment stays disabled. |
| `401 Token expired` | Set `LITELLM_API_KEY_HELPER`. |
| No models with gcloud auth | Verify `gcloud auth application-default login` has been run or set `GOOGLE_APPLICATION_CREDENTIALS` to an `authorized_user` ADC file |
| Enterprise SSO waits for token insertion | The proxy returned 404/405 for `/sso/cli/start`, so Pi used the legacy flow — upgrade LiteLLM or paste the UI token |
| Enterprise CLI SSO start/poll fails | Check the proxy logs and verify `/sso/cli/start` and `/sso/cli/poll/{login_id}` are reachable; only 404/405 falls back to legacy login |
| Enterprise SSO login shows "virtual key generation failed" | The LiteLLM instance may lack a database (`/key/generate` requires one), your user account may lack key-generation permission, or the request timed out; the JWT is used directly as a fallback |
| Enterprise SSO token prompt fails with "SSO token is required" | The token field was left empty — paste the token copied from the LiteLLM UI |
| MCP tools not showing | Verify the proxy exposes `/mcp-rest/tools/list` and open `/model` after fixing the proxy |
| Skills not affecting prompts | Verify the proxy exposes `/claude-code/marketplace.json` or `/v1/skills` and returns enabled skills |

### `No models available` at startup

Pi 0.84.0 and later can finish startup before the provider availability snapshot is written, so the
initial model pick sees nothing even though discovery succeeded. The warning is intermittent and a
restart usually clears it. Nothing is wrong with the catalog: `/model` in the same session still
lists every discovered model, and picking one there fixes that session.

Scoping models keeps selection off that path, because Pi resolves the scope with its own awaited
availability pass before it picks a model. In `~/.pi/agent/settings.json`:

```json
{
  "enabledModels": ["litellm/*"]
}
```

Your `defaultProvider` and `defaultModel` still win as long as they match a pattern. `/model` then
opens on the scoped list, with a toggle inside the picker for the full catalog.

The extension cannot set this for you — Pi reads `enabledModels` before extensions activate.

## License

MIT — see [LICENSE](./LICENSE).
