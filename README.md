# pi-provider-litellm

LiteLLM proxy native Provider extension for [Pi](https://pi.dev). Pi 0.99.2+ is required; on older Pi, install `npm:pi-provider-litellm@3.4.0`.

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

Or for a configured alias provider (e.g. `team-a`):

```
/login team-a
```

To configure an API key, run `/login`, choose `Sign in with an API key`, then choose `LiteLLM API key` (or `<Provider Name> API key`). With `/login litellm`, choose `Sign in with an API key` directly; similarly, run `/login <alias>` to sign in to any configured alias directly.

You'll be prompted for the base URL and API key. Credentials are persisted to `~/.pi/agent/auth.json`.

If a proxy URL is already known — from `providers.<name>.baseUrl`, `LITELLM_PROVIDER_<NAME>_BASE_URL`, `LITELLM_BASE_URL`, or a previous login — pi offers it as the first option instead of asking you to retype it. Pick `Enter a different URL…` to point at another proxy.

#### Enterprise SSO login

If your LiteLLM proxy supports SSO/OAuth authentication, Pi selects its supported browser login flow automatically:

1. Run `/login litellm` inside pi and select `Sign in with LiteLLM SSO`
2. Confirm the offered proxy URL, or enter one if this is the first login
3. Complete sign-in in the browser Pi opens, then return to Pi

Without an `oidc` setting (see [Direct OIDC login](#direct-oidc-login)), Pi first checks `/.well-known/litellm-cli-auth`. A proxy exposing the supported CLI-auth contract uses authorization code + PKCE (`S256`) with a temporary `127.0.0.1` callback; the browser must run on the same machine as Pi. Advertised endpoints must stay on the proxy's origin, and discovery, registration, and token requests do not follow HTTP redirects. Invalid discovery stops login.

Pi stores PKCE access and refresh tokens in `~/.pi/agent/auth.json` with file mode `0600`, refreshes automatically, and saves each rotated token pair. Temporary network failures or HTTP 429/5xx responses allow reuse of the existing access token only until its exact expiry. Rejected or revoked refresh credentials require `/login litellm` again.

If discovery returns 404, Pi uses `/sso/cli/start`: confirm the verification code in the browser, then select a team in Pi if prompted. Pi polls for the short-lived CLI token and uses its advertised lifetime, falling back to `LITELLM_CLI_JWT_EXPIRATION_HOURS` or LiteLLM's 24-hour default on older proxies. Only LiteLLM versions that gate this older CLI SSO flow need `EXPERIMENTAL_UI_LOGIN=True` on the proxy.

If `/sso/cli/start` returns 404 or 405, copy the token from the LiteLLM UI and paste it into Pi, with an optional virtual-key exchange. Pi reads JWT expiry claims and prompts for login when re-authentication is required.

Tokens are stored in Pi's local file, not an OS keychain. `/logout litellm` deletes the local credential only; logging out or signing in again does not revoke the previous session on the proxy because Pi has no provider revocation hook.

##### Direct OIDC login

For proxies running LiteLLM JWT auth (`enable_jwt_auth`) against your OpenID Connect identity provider (IdP), Pi can sign in with the IdP directly as a registered public client and present the IdP's `id_token` to the proxy as the bearer token. The proxy mints nothing and no virtual key is created; the IdP stays the session authority.

Prerequisites:

- The proxy's JWT auth trusts the IdP's issuer and accepts the client ID as an audience.
- The IdP has a public client (no client secret) registered for authorization code + PKCE with the loopback redirect `http://127.0.0.1:<port>/callback`.
- The browser runs on the same machine as Pi.

Configure the default `litellm` provider in the global `~/.pi/agent/settings.json`. Project settings are never read for this, so a cloned repository cannot point login at its own IdP.

```json
{ "litellm": { "providers": { "litellm": {
  "baseUrl": "https://litellm.example.com",
  "oidc": {
    "issuer": "https://idp.example.com",
    "clientId": "example-client-id",
    "scope": "openid",
    "redirectPorts": [8400, 8401]
  }
} } } }
```

- `issuer` (required): absolute `https` URL with no credentials, query, or fragment.
- `clientId` (required): the public client's ID.
- `scope` (default `"openid"`): must include `openid`.
- `redirectPorts` (optional): ports tried in order, for IdPs that only accept exact pre-registered redirect URIs. Without it the OS assigns a free port.

With `oidc` set, `Sign in with LiteLLM SSO` confirms the proxy URL and then runs only this flow: Pi reads `<issuer>/.well-known/openid-configuration`, opens the IdP's authorization page, and exchanges the code for tokens. Login sends nothing to the proxy, and IdP requests never carry `LITELLM_HEADERS` or provider `headers`. An invalid `oidc` setting fails login with a message naming the field; it never falls back to the LiteLLM flows.

The `id_token` is the bearer token. Pi checks its issuer, audience, authorized party (when present), subject, expiry, and nonce but not its signature; the proxy verifies the signature against the IdP's keys. Pi refreshes it with the refresh token, so the IdP must return a new `id_token` from the refresh grant. Some IdPs only issue a refresh token when the scope includes `offline_access`; without one, run `/login litellm` again when the `id_token` expires.

Pi never derives the IdP from the proxy: LiteLLM's own `/.well-known/openid-configuration`, `oauth-authorization-server`, and `oauth-protected-resource` documents describe LiteLLM's authorization server, not the IdP that signs the JWTs it accepts. Zero-config IdP discovery from the proxy via RFC 9728 is out of scope until LiteLLM publishes it ([BerriAI/litellm#41135](https://github.com/BerriAI/litellm/issues/41135)).

### Option B — environment variables

#### Single provider

```bash
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_API_KEY="sk-..."
```

Stored pi credentials for `litellm` take precedence over `LITELLM_API_KEY`; the environment key is used when no saved credential exists. `LITELLM_BASE_URL` is used when no saved login base URL exists. Chat Completions and Responses models use the proxy root plus `/v1`; native Messages uses the proxy root directly.

#### Multi-provider environment variables

You can configure additional LiteLLM provider aliases directly through environment variables without editing `settings.json`.

##### Per-provider environment variables (`LITELLM_PROVIDER_<NAME>_*`)

Use the prefix `LITELLM_PROVIDER_<NAME>_` where `<NAME>` is an uppercase identifier (e.g., `TEAM_A`, `CORP_EAST`):

| Variable | Effect |
|---|---|
| `LITELLM_PROVIDER_<NAME>_BASE_URL` | LiteLLM proxy base URL for this provider (e.g. `https://team-a.example.com`) |
| `LITELLM_PROVIDER_<NAME>_API_KEY` | API key / bearer token for this provider |
| `LITELLM_PROVIDER_<NAME>_API_KEY_HELPER` | Shell command that prints a fresh token for this provider |
| `LITELLM_PROVIDER_<NAME>_HEADERS` | JSON string of extra request headers for this provider |
| `LITELLM_PROVIDER_<NAME>_DISPLAY_NAME` | Custom display name shown in Pi UI |
| `LITELLM_PROVIDER_<NAME>_NAME` | Explicit provider ID override (e.g. `custom-name`) |
| `LITELLM_PROVIDER_<NAME>_ALLOW_INSECURE_HTTP` | Set `"1"` or `"true"` to permit plaintext HTTP for non-loopback hosts |
| `LITELLM_PROVIDER_<NAME>_USE_GCLOUD_AUTH` | Set `"1"` or `"true"` to enable Google ADC token authentication for this provider |
| `LITELLM_PROVIDER_<NAME>_ENABLE_OAUTH` | Set `"1"` or `"true"` / `"0"` or `"false"` to enable or disable LiteLLM SSO/OAuth browser login |
| `LITELLM_PROVIDER_<NAME>_OIDC` | JSON string or OpenID Connect issuer URL for direct IdP OIDC login |

##### Bulk provider configuration

- **`LITELLM_PROVIDERS`**: Canonical comma- or whitespace-separated list of provider IDs to register (e.g., `export LITELLM_PROVIDERS="team-a, team-b, corp-east"`).
- **`LITELLM_PROVIDERS_JSON`**: Structured JSON string matching the `litellm.providers` schema in `settings.json` (e.g., `export LITELLM_PROVIDERS_JSON='{"team-a": {"baseUrl": "https://team-a.example.com", "apiKey": "sk-team-a"}}'`).

##### Provider ID naming rules

- **Default naming:** The uppercase token `<NAME>` in `LITELLM_PROVIDER_<NAME>_*` is automatically converted to lowercase kebab-case (e.g., `TEAM_A` becomes `team-a`, `CORP_EAST` becomes `corp-east`, and `DEV` becomes `dev`).
- **Explicit override:** Set `LITELLM_PROVIDER_<NAME>_NAME` to specify an explicit provider ID (e.g., `LITELLM_PROVIDER_TEAM_A_NAME="custom-team"`).
- **Canonical IDs:** Provider IDs specified in `LITELLM_PROVIDERS` or keys in `LITELLM_PROVIDERS_JSON` preserve their declared IDs.
- **Validation:** Provider IDs must match `/^[a-z0-9][a-z0-9_-]{0,63}$/` (1–64 characters, lowercase alphanumeric with `-` or `_`, starting with an alphanumeric character).
- **Reserved names:** Tool and command names (`mcp`, `skills`, `codemode`, `tool_search`, and `litellm` as an alias) cannot be used as secondary provider IDs.

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
| `baseUrl` | `LITELLM_BASE_URL` for `litellm`; required for aliases | LiteLLM proxy URL, with or without `/v1`. Must be a full `http`/`https` URL; a provider with no resolvable base URL exposes no models (see [Model host enforcement](#model-host-enforcement)) |
| `apiKey` | `LITELLM_API_KEY_HELPER`/`LITELLM_API_KEY` for `litellm`; required for aliases | Pi config value for this provider's key. Use `$ENV_VAR`, `${ENV_VAR}`, `!command`, or a literal key. Escape a literal `$` as `$$`. |
| `headers` | `$LITELLM_HEADERS` for `litellm`; unset for aliases | JSON string env reference or inline object of request headers |
| `displayName` | `LITELLM_DISPLAY_NAME` / `"LiteLLM"` for `litellm`; alias name for aliases | Label shown in Pi UI |
| `enabled` | `true` | Set `false` to skip or disable an alias |
| `oidc` | unset | Sign in directly with an OpenID Connect identity provider instead of the LiteLLM-hosted flows; see [Direct OIDC login](#direct-oidc-login) |
| `allowInsecureHttp` | `false` | Set `true` to permit plaintext HTTP for this provider, for example `http://host.docker.internal`. Credentials and request data will not be encrypted. Loopback HTTP works without this setting. |
| `useGcloudTokenAuth` | `true` for `litellm`; `false` for aliases | Set `true` to enable Google Application Default Credentials (ADC) token authentication for this provider. |
| `enableOAuth` | `true` for `litellm`; `false` for aliases | Set `true` to enable LiteLLM SSO/OAuth browser login for this provider. |

#### Configuration & credential precedence

- **Provider definition merging precedence:**
  When discovering and configuring provider aliases from multiple sources, definitions are merged in the following order:
  1. Base provider definitions from `LITELLM_PROVIDERS_JSON`
  2. Discovered environment variables (`LITELLM_PROVIDER_<NAME>_*` and canonical list in `LITELLM_PROVIDERS`)
  3. Disk settings from `~/.pi/agent/settings.json` under `litellm.providers`. Disk settings override environment definitions, and setting `"enabled": false` removes or skips the provider.

- **Credential resolution precedence:**
  When authenticating requests for any provider:
  1. Saved interactive credentials in `~/.pi/agent/auth.json` (from `/login <provider>` or OAuth SSO)
  2. Google Application Default Credentials (when `useGcloudTokenAuth` / `_USE_GCLOUD_AUTH` / `LITELLM_GCLOUD_TOKEN_AUTH` is active)
  3. Configured `apiKey` from settings (`settings.json` or `LITELLM_PROVIDERS_JSON`, including `$ENV_VAR` and `!command`)
  4. Provider-scoped API key helper (`LITELLM_PROVIDER_<NAME>_API_KEY_HELPER`)
  5. Provider-scoped API key (`LITELLM_PROVIDER_<NAME>_API_KEY`)
  6. For the default `litellm` provider only: fallback to global `LITELLM_API_KEY_HELPER` then `LITELLM_API_KEY`.

### Optional LiteLLM features

LiteLLM Skills and MCP integration are enabled by default. Disable either feature, or choose how MCP tools reach the model, globally in `~/.pi/agent/settings.json`:

```json
{
  "litellm": {
    "skills": {
      "enabled": false
    },
    "mcp": {
      "enabled": true,
      "exposure": "deferred",
      "toolExposure": {
        "*delete*": "hidden",
        "github-search_code": "direct"
      }
    }
  }
}
```

Setting `skills.enabled` to `false` disables the Skills Gateway management tools, skill fetching, and system-prompt injection. Setting `mcp.enabled` to `false` stops the extension from registering the proxy's MCP server. Restart Pi after changing these settings.

`mcp.exposure` and `mcp.toolExposure` are passed to Pi unchanged and mean what they mean in a Pi `mcp.json` server entry: `codemode` (Pi's default when `exposure` is unset; callable from Pi's `codemode` scripts), `deferred` (declared to the model once Pi's `tool_search` loads it), `direct` (declared on every request), or `hidden`. `toolExposure` keys are tool names as LiteLLM's `/mcp` endpoint offers them, which LiteLLM prefixes with the server name, for example `github-search_code`; `*` matches any characters. Pi validates both settings, and an unusable value is reported once.

Treat the configured LiteLLM proxy as trusted: Skills can add instructions to the system prompt, and MCP can expose tools the agent may call. Disable these integrations when the proxy, its administrators, or its configured content are not fully trusted.

## Use

```
/model
```

To refresh catalogs on demand, run `/litellm-refresh` for every configured LiteLLM provider, or `/litellm-refresh <provider>` for one alias. It works with `PI_OFFLINE` set, which stops Pi's own `/model` refresh from contacting the network. It contacts only the configured proxies: `LITELLM_OFFLINE=1` and `LITELLM_DISCOVERY_TIMEOUT_MS=0` still disable it, and models.dev enrichment stays off under `PI_OFFLINE`. Each provider reports its model count, the failure, or missing credentials. MCP tools are not part of this refresh; run `/mcp reconnect <provider>` to reconnect a provider's MCP server.

Every request routed through a configured LiteLLM provider includes Pi's canonical session ID in the
`x-litellm-session-id` header. This groups Chat Completions, Responses, and native Messages requests in LiteLLM without
adding transport-specific fields to request bodies.

## Model transport

A `/model/info` route group uses native Anthropic `/v1/messages` only when every deployment explicitly reports Chat mode, identifies a Claude backend through an Anthropic-, Bedrock-, or Vertex-capable adapter, and every declared backend identity resolves the same Anthropic compatibility policy. A `claude-<name>-<major>-<minor>` backend id resolves through its `claude-<name>-<major>` entry when the exact id is absent; an unknown generation stays on Chat Completions. When `supported_endpoints` is a list, it must include `/v1/messages`; an explicit endpoint list without it keeps the group on Chat Completions. A `null` value, which LiteLLM v1.100 and later report for models absent from their model map, counts as absent. An unresolved routing or base-model identity denies native Messages rather than being discarded from the unanimity check. Mixed-generation Claude groups can use Messages when their serializer policies match, with reasoning levels restricted to their common supported set. Groups with incompatible policies, mixed families, or unknown backend evidence remain on Chat Completions; unanimous explicit Responses mode takes precedence. Routes discovered through `/health` never select native Messages: Messages candidates are downgraded to Chat Completions with reasoning controls closed, while correlated Responses detail remains on Responses. Evidence-free `/v1/models` and wholly health-only `/health` groups take transport and presentation metadata from the bounded Pi catalog lookup: `openai-responses` when the catalog says so, otherwise Chat Completions. They expose no speculative reasoning selector and never select native Messages. Catalog identity such as Amazon Bedrock remains separate from the selected wire protocol.

Native Messages authenticates with `x-api-key`; every transport carries the `x-litellm-session-id` header described above. Strict JSON-schema tools are enabled separately from Messages transport and thinking compatibility: every deployment must affirm support for the actual route/API. Generic `supports_response_schema` and `supports_native_structured_output` metadata does not grant strict tools. Only deployments that LiteLLM reports with the `anthropic` adapter carry that evidence today. Claude served through Bedrock or Vertex therefore uses ordinary tool serialization, and a tool that requires strict sampling fails before the request is sent.

## Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `LITELLM_API_KEY_HELPER` | unset | Command that prints a fresh LiteLLM bearer token. Takes precedence over `LITELLM_API_KEY`. The extension runs it while resolving request auth, and Pi's per-request auth path is uncached, so rotating/short-lived tokens stay fresh. |
| `LITELLM_DISPLAY_NAME` | `LiteLLM` | Custom display name for the default `litellm` provider in the Pi UI. |
| `LITELLM_HEADERS` | unset | JSON object of extra headers sent to LiteLLM provider, discovery, MCP, and Skills Gateway requests. Provider aliases can use it with `"headers": "$LITELLM_HEADERS"`. |
| `LITELLM_PROVIDERS` | unset | Canonical comma- or whitespace-separated list of provider IDs to register. |
| `LITELLM_PROVIDERS_JSON` | unset | JSON string of provider configuration objects matching the `litellm.providers` settings schema. |
| `LITELLM_PROVIDER_<NAME>_*` | unset | Per-provider environment variables for base URL, API key, headers, display name, OAuth, OIDC, and Google auth. See [Multi-provider environment variables](#multi-provider-environment-variables). |
| `LITELLM_GCLOUD_TOKEN_AUTH` | unset | If set to a non-empty value other than `0`, use Google Application Default Credentials as the LiteLLM bearer token source. This takes precedence over `LITELLM_API_KEY_HELPER` and `LITELLM_API_KEY` when no stored `/login litellm` credential exists. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google default ADC path | Optional path to an ADC JSON file used by `LITELLM_GCLOUD_TOKEN_AUTH`. If unset, the extension checks the default gcloud ADC locations. |
| `LITELLM_OFFLINE` | unset | If `1`, disable all model and MCP discovery, including post-login discovery; use cached models only when their stored canonical proxy root exactly matches the active credential root, including any path prefix. URL-standard host casing and default ports are canonicalized, but paths remain case-sensitive. |
| `PI_OFFLINE` | unset | Pi documents it as disabling automatic network activity, including model catalog refreshes. Through Pi 1.0.3, any set value (including `0` or an empty string) disables startup model discovery and `/model` network refreshes. Run `/litellm-refresh`, or unset it, to discover; cached models can still be used offline. |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Background and explicit discovery fetch timeout in ms; `0` disables automatic discovery |
| `LITELLM_CLI_JWT_EXPIRATION_HOURS` | `24` | CLI SSO token lifetime fallback for older proxies whose poll response omits `expires_in`; mirror a non-default proxy setting locally |
| `LITELLM_VERBOSE_DISCOVERY` | unset | If `1`, enable progress messages during model discovery (login, refresh, startup), including startup skip reasons and defaulted `/model/info` context windows. Progress messages are off by default |
| `LITELLM_DEFAULT_CONTEXT_WINDOW` | `128000` | Context window assumed when neither LiteLLM's `/model/info` nor the catalog reports `max_input_tokens`. Set a positive integer for proxies whose custom aliases carry no metadata; an unset or unusable value keeps 128K |
| `LITELLM_MODELS_DEV` | unset | If `1`, enrich discovered metadata (limits, prices, effort lists) from models.dev, cached for 28 days in `litellm-models-dev.json`. Off by default because LiteLLM's `/model/info` is authoritative; use it when LiteLLM's model map lacks a model's metadata |

Only use a trusted `LITELLM_API_KEY_HELPER` or `!command`. Prefer an absolute executable path, keep secrets out of command arguments, and print only the token to stdout without logging it to stderr.

With `LITELLM_VERBOSE_DISCOVERY=1`, `/model/info` discovery reports published routes whose context window uses the fallback assumption (including wildcard expansions) on one line with the selected value, a count, and a sample of escaped route names. Each route is reported once per process. Explicit or authoritative catalog limits do not trigger it, even when numerically equal to the default. Set `model_info.max_input_tokens` on every deployment in the route to provide real limits; this diagnostic does not change discovery limits or routing.

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

LiteLLM serves the MCP servers it gateways over streamable HTTP at `/mcp`. For each configured provider, the extension registers that endpoint as a Pi MCP server named after the provider (`litellm`, or the alias name), and Pi's built-in MCP client connects to it, lists its tools, and calls them. Tools are named `mcp__<provider>__<tool>` and appear in `/mcp`, where the server's connection state and errors are shown.

The registration does not copy the provider's API key or token. It uses Pi's `auth: { provider }` setting, so Pi sends the provider's current credential as `Authorization: Bearer` on every request, and token refreshes, Google ADC, and key helpers apply without registering again. Custom headers (`LITELLM_HEADERS`, or an alias's `headers`) are registered as configured, after `$` and a leading `!` are escaped so Pi does not expand them a second time. Pi sends provider credentials only over https or to a loopback host, so a proxy reached over plain http elsewhere (`allowInsecureHttp`) gets no MCP server, and the refusal is reported once.

The extension registers at startup and checks again before each agent turn. `/login litellm` disconnects the server as soon as it starts, before the new credential is stored, and the next turn connects again with the new credential and proxy root; a logout that leaves no root removes the server. `LITELLM_OFFLINE=1`, `LITELLM_DISCOVERY_TIMEOUT_MS=0`, and `PI_OFFLINE` register nothing. A server with the same name in your own `mcp.json` takes precedence over the extension's.

Pi reads the provider's credential when it sends each request, not when the server was registered. If another Pi process stores a login for a different proxy, this session's open connection can send that credential to the previous proxy until the next turn moves the server.

Earlier versions discovered tools through `/mcp-rest/tools/list` and registered them as `mcp_<server>_<tool>_<hash>`. Those names and the `<server>/<tool>` form of `toolExposure` keys no longer apply, and the `~/.pi/agent/litellm-mcp-pauses/` directory is no longer used.

Tools are only as trustworthy as the MCP servers behind your proxy: their descriptions and results are text the model reads.

## LiteLLM Skill Hub

If your LiteLLM proxy exposes `/claude-code/marketplace.json`, enabled skills are fetched before each agent turn and appended to the system prompt as a `litellm_skills` section. The extension falls back to the legacy `/v1/skills` Skills Gateway path when Skill Hub is unavailable. It also registers Pi tools for basic skill management:

- `litellm_skill_list`
- `litellm_skill_create`
- `litellm_skill_delete`

`litellm_skill_create` needs Skill Hub `sourceJson` metadata and a proxy that exposes `/claude-code/plugins`. It does not create skills from code: LiteLLM's `POST /v1/skills` is Anthropic's multipart Skills API.

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

Dynamic catalogs are persisted by Pi in `~/.pi/agent/models-store.json`. Credentials remain in `~/.pi/agent/auth.json`. Legacy `litellm-models.json` model caches are ignored and never deleted. `litellm-models-dev.json` is the models.dev cache and is refreshed in place.

LiteLLM's `/model/info` is the metadata authority, so models.dev enrichment is an opt-in escape hatch. With `LITELLM_MODELS_DEV=1`, for every genuine `/model/info` row, including deployment details fetched through `/health`, the extension requests `https://models.dev/api.json` for enrichment and caches the result for 28 days in `~/.pi/agent/litellm-models-dev.json`. Without it, discovery neither requests models.dev nor reads that cache; Pi's own catalog still applies. Health-only entries without deployment details use the bounded Pi catalog lookup without models.dev enrichment. `PI_OFFLINE` suppresses activation-time discovery and the models.dev request. `LITELLM_OFFLINE=1` also disables LiteLLM discovery; direct discovery callers use only an existing models.dev cache and do not refresh it.

Opening `/model` refreshes configured provider catalogs in the background using Pi's native model lifecycle when network discovery is enabled.

Pi documents `PI_OFFLINE` and `--offline` as disabling automatic network activity, including model catalog refreshes. Pi 0.87.1 adopted that wording to resolve [earendil-works/pi#8684](https://github.com/earendil-works/pi/issues/8684), without changing the behavior. Pi still checks only whether the variable is set, so `PI_OFFLINE=0` and an empty value also disable model refreshes. With `LITELLM_VERBOSE_DISCOVERY=1`, disabled startup discovery reports its reason, for example `startup discovery skipped (PI_OFFLINE)`. To discover models while keeping `PI_OFFLINE`, run `/litellm-refresh`. To allow all discovery for one invocation, use `env -u PI_OFFLINE pi` and do not pass `--offline`. If you only want to disable Pi's version check, use `PI_SKIP_VERSION_CHECK=1` instead; it does not apply offline mode's other network restrictions. See [#155](https://github.com/balcsida/pi-provider-litellm/issues/155).

### Model host enforcement

Models this provider dispatches are sent to the root resolved from the active credential. The extension hides a model from `/model` when its stored root, including any path prefix, does not match that credential, and its dispatch-time guard rejects stale, malformed, placeholder, and unsupported models before a request. For accepted models, the request URL is re-derived from the credential's proxy root instead of trusting a configured model URL. Explicit `allowInsecureHttp` settings continue to apply to this validation. Path prefixes are part of the root and remain case-sensitive.

Catalog filtering and dispatch are separate. Choosing a model by ID or restoring it from a session can skip filtering. Pi 0.84 routes such a model to the provider only when the provider's current catalog already contains the same `api`. The native `Provider` contract has no separate protocol-capability declaration. If the catalog contains no model for that API, Pi uses its global API implementation instead, bypassing this extension's dispatch-time host guard. The extension deliberately returns an ordinary model array rather than falsifying its contents through an overridden Array method.

To keep that fallback safe, resolved auth carries `baseUrl` set to the credential's proxy root. Pi applies it to every request path — the provider guard and the global fallback alike — so a stale, placeholder, or attacker-supplied model `baseUrl` is replaced with the credential root before the request leaves the process. The credential can only ever reach the host it was issued for. A model whose `api` is absent from the catalog may still fail to complete on the fallback path (the global implementation builds a slightly different URL), but it cannot exfiltrate the credential. When no usable root resolves, auth carries no `baseUrl` and the provider guard rejects the request outright.

Opening `/model` against the active proxy repopulates the protocols discovery actually selects, which is the supported way to use a model whose `api` is currently absent from the catalog.

A model is hidden when:

- no base URL resolves from settings or credentials
- the base URL is invalid or remains the `https://litellm.example.com` placeholder
- its stored root differs from the active credential root, such as after switching proxies or path-scoped tenants
- it declares an API this extension does not implement

Each distinct availability diagnostic is written once per session on stderr. `LITELLM_OFFLINE=1` does not recover a root mismatch: refresh online against the active proxy first, then return to offline use.

### Protocols and prompt caching

Discovery selects native Messages for proven Claude groups under the [Model transport](#model-transport) rules, Responses for OpenAI-family backends, and Chat Completions for other OpenAI-compatible backends. Azure deployments stay on Chat Completions unless an explicit Responses mode or a `supported_endpoints` list containing `/v1/responses` authorizes Responses; `api_version` alone does not prove support. Azure AI (`azure_ai/` prefix or `custom_llm_provider: azure_ai`) deployments additionally depend on the proxy version. On LiteLLM before v1.103.0, or when the version is unknown, they ignore a `supported_endpoints` list that contains `/v1/responses` and need an explicit `mode: "responses"`: those releases have no native Responses config for that provider, so they bridge `/v1/responses` through Chat Completions, and the bridge fails streaming with `list index out of range` on Azure's empty-choices chunks (BerriAI/litellm#34455) while the endpoint list, copied from the model cost map, still advertises `/v1/responses`. From v1.103.0 the list is honored again. Discovery reads the version from the `x-litellm-version` header of one extra request, `GET /v1/responses/resp_version_probe`, sent only when such a deployment is published; LiteLLM answers it with an error and calls no model. A pre-release counts as older than the release it names, and a proxy that omits the header keeps these deployments on Chat Completions. Azure is detected from the adapter field, an `azure/` or `azure_ai/` model prefix, or the reported provider; other adapters follow the backend family because LiteLLM bridges `/v1/responses` to Chat Completions when the provider has no native Responses config (`litellm/responses/main.py`, `_bridges_to_chat_completions`). The family comes from the model id, not from LiteLLM's `openai/`, `openai_like/` or `custom_openai/` adapter prefix, so `openai/qwen3` stays on Chat Completions while `openai/gpt-5.5` uses Responses; a deployment opts into Responses with `mode: "responses"` or a `supported_endpoints` list containing `/v1/responses`. A route whose model prefix and `custom_llm_provider` name different providers is treated as unidentified and stays on Chat Completions, so it also does not get the local 128-tool preflight. When `/model/info` supplies `supported_endpoints`, that list takes precedence, and `mode: "responses"` remains an explicit Responses signal. An evidence-free fallback entry—a bare `/v1/models` id or a `/health` entry with no detail row—takes its protocol from the Pi catalog entry for its id; both evidence-free fallback paths also inherit presentation metadata from the bounded catalog lookup. An unknown id stays on Chat Completions. The route name alone authorizes nothing. A concrete id expanded from wildcard `/model/info` routes inherits the deployment evidence of the wildcard route LiteLLM would select for it, with the requested id substituted into the row's `model` the way LiteLLM serves it; the id is omitted when that selected route is not a published chat-style route. Chat Completions and Responses use `<root>/v1`; native Messages uses the proxy root directly. All three protocols pass through the provider's host guard. When an evidence-free cached `(no metadata)` entry can be enriched from the Pi catalog, it is restored on Responses only for a catalog Responses model and otherwise on Chat Completions, never Anthropic Messages. A model supplied only through `models.json` whose `api` this provider does not implement takes Pi's global API fallback rather than this provider's host guard; the credential-root pin (see [Model host enforcement](#model-host-enforcement)) still keeps the credential on the active proxy.

`cacheControlFormat: "anthropic"` applies only to Chat Completions, where Pi adds Anthropic `cache_control` markers. The Responses transport has a different compatibility type and uses native `prompt_cache_key` (plus supported retention fields), so Responses models must not receive `cacheControlFormat`. Freshly discovered OpenAI-family models forced onto Chat Completions are rejected locally when a request contains more than 128 tools; route them through Responses or reduce enabled extensions. Moonshot request-side reasoning suppression follows LiteLLM's routing provider, not `model_info.base_model`; response-side `<think>` normalization follows discovered Kimi-family evidence. Opaque aliases retain display normalization without sending Moonshot-only parameters to Azure- or Bedrock-hosted Kimi models. Strict tool-message repair requires every deployment to identify Kimi; Gemini effort normalization likewise requires consistent backend-family evidence.

Discovery policy is versioned in Pi's persisted model catalog. After upgrading, an online refresh replaces legacy entries. Offline legacy entries remain usable without a diagnostic; the extension reports the discovery-version mismatch once only when an attempted online refresh fails. To force rediscovery or roll back across this policy change, delete the `litellm` entry from `~/.pi/agent/models-store.json` (or delete the file) and refresh once online.

### Deployment groups and metadata authority

LiteLLM may load-balance one public `model_name` across deployments with different backends or model versions. The extension reduces `/model/info` rows conservatively before publishing one Pi model. A route group that mixes a chat-style deployment with an explicitly incompatible mode such as embedding is withheld entirely and reported with a bounded diagnostic.

- Responses is selected only when every deployment supports it under the protocol rules above; a deployment that requires Chat keeps the group on Chat.
- Vision and reasoning are advertised only when every routable deployment resolves them as supported. An explicit `supports_reasoning: false` suppresses all thinking controls, even when catalog or router effort metadata exists. Catalog thinking controls are intersected per level: a level is exposed only when every deployment maps it to the same value, while disagreement or absence becomes an explicit denial. Catalog maps are not complete effort lists, so absence has one exception: an omitted `minimal`, `low`, `medium` or `high` counts as Pi's default spelling of that level. An omitted `off`, `xhigh` or `max` still differs from an explicit mapping. Reasoning selectors additionally require an accepted wire carrier on every deployment. Public effort lists are intersected, explicit LiteLLM denials win, and `xhigh`/`max` need unanimous explicit support. Null or absent standard-effort flags have no opinion. A deployment that declares `model_info.reasoning_effort_levels` answers every level from that list, ahead of its per-level flags, as LiteLLM itself reads it.
- Context and output limits use the minimum resolved value across deployments. Pi catalogue lookup prefers the deployment's provider-specific entry before falling back to the generic vendor entry; Azure and Azure AI therefore retain Azure-specific limits rather than being capped by a different OpenAI catalogue entry. When models.dev supplies a sparse vendor record, missing fields still use the provider-specific Pi catalogue.
- Each displayed base-price field uses the maximum only when every deployment resolves that field; unresolved fields remain zero and the model name is suffixed with ` (incomplete metadata)`.
- When complete base pricing comes from the catalog under unanimous catalog authority, tiered pricing is the conservative worst-case envelope. Its usable finite non-negative thresholds are the sorted union of every deployment's thresholds, and each price field at each interval is the maximum applicable rate across all deployments, floored by each deployment's base rate. Differing tier breakpoints are therefore preserved rather than treated as incompatible. An explicit `/model/info` base-price field replaces catalog pricing for that field at every threshold; catalog tiers remain for unaffected fields and are omitted only when every base-price field is explicit.
- Reduced route groups omit tier ladders when complete base-pricing or catalog authority is unavailable. Wildcard expansion retains every known tier from matching groups even when another match has incomplete metadata: suppressing a complete sibling's higher tier could understate the known worst-case rate. The ` (incomplete metadata)` marker remains because the resulting envelope is only as complete as the available ladders.
- Wildcard routes expand through `/v1/models`; dropped exact or wildcard groups suppress matching expanded ids.
- Expanded ids inherit conservative metadata bounds from every matching wildcard group, including the same union-threshold, maximum-applicable-rate tier envelope. Protocol and backend-family evidence follow the most-specific route LiteLLM selects, with every deployment of that selected route voting together.
- Catalog metadata is accepted only from unanimous deployment identities resolved from `model_info.base_model` first and `litellm_params.model` second; the public `model_name` route is never backend authority. `litellm_params.custom_llm_provider` is provider evidence for the selected backend model: a non-generic provider that conflicts with the model prefix withholds identity, while generic transport adapters are ignored. Recognized provider prefixes remain identity evidence even when the named model is absent from that provider's catalog. Different concrete catalog models conflict even within one provider; an unresolved Azure deployment name may still use its resolved `base_model`. Conflicts across a group trigger the bounded ambiguous-authority diagnostic; ambiguous groups are not matched across all Pi provider catalogs. Cross-host Claude routes spanning providers such as Vertex and Bedrock may intentionally trigger this withholding: explicit router metadata remains usable, but the route does not borrow either host's catalog metadata.
- `/v1/models` and `/health` do not provide deployment-level backend identity. For `/v1/models`, an unqualified ID is resolved only within an explicitly recognized `owned_by` provider or through the bounded alias rules documented here (currently Anthropic aliases), and is never searched across every Pi provider catalog. Both `/v1/models` fallback entries and `/health` entries without deployment detail take their protocol from the matching Pi catalog model: `openai-responses` selects Responses, and every other or missing catalog API selects Chat. Wholly health-only groups also inherit presentation metadata from that bounded Pi catalog lookup; a bare row mixed with deployment details grants no catalog authority to the group. A route name substituted for an unreadable deployment `model_name` also denies router-reported thinking levels. Unresolved evidence-free entries retain the ` (no metadata)` marker; groups containing deployment details remain permanently incomplete when their evidence is insufficient.

The ` (no metadata)` suffix is reserved for unresolved, evidence-free entries mapped by the full `/v1/models` fallback or wholly health-only `/health` groups. During wildcard expansion after a successful `/model/info` response, IDs that match no surviving wildcard route are discarded. A matched wildcard expansion inherits its matching groups' authority and is either unmarked or marked ` (incomplete metadata)`, never ` (no metadata)`. A recognized `owned_by` provider that conflicts with a provider-qualified model ID instead receives ` (incomplete metadata)`, permanently withholding cache enrichment rather than falling through to the ID prefix. Only an unchanged evidence-free sentinel shape is eligible for bounded catalog enrichment on a later cache read, when it also refreshes provider compatibility metadata from the model ID, adopts Responses only for a catalog `openai-responses` model, and otherwise uses Chat.

The ` (incomplete metadata)` suffix marks reduced `/model/info` groups, incomplete matched wildcard expansions, or `/health` groups with deployment details and permanently prevents route-name cache enrichment. It means at least one metadata field is unknown, including cache pricing that the proxy omitted; known input/output prices may still be shown alongside the suffix.

### Reasoning controls

Reasoning levels require deployment evidence for the parameter Pi sends: `reasoning_effort` for effort levels or `thinking` for a native thinking switch. Public route names never authorize these controls.

LiteLLM omits `supported_openai_params` for a deployment its model map does not describe. Such a deployment keeps the `reasoning_effort` carrier when its `model_info` explicitly sets `supports_reasoning: true`, which is the operator's opt-in. A `supported_openai_params` list that omits `reasoning_effort` still denies effort levels unless `litellm_params.allowed_openai_params` adds it. Kimi and DeepSeek generations are excluded because their contracts name their own carriers. Evidence-free fallback models retain catalog presentation metadata but expose no speculative selector.

GPT-5.5 and later reject `reasoning_effort` together with function tools on `/v1/chat/completions`. On a Chat route where any deployment's `model_info.base_model` or `litellm_params.model` names such a backend, tool requests turn reasoning off instead of failing or being answered by a router fallback: `reasoning_effort` is set to the model's declared off value, or removed when it declares none. GPT-6 and later fall back to their default effort when the field is omitted, so they get `none` either way. A model with no backend evidence falls back to its route name. Requests without tools keep their reasoning level. A model that cannot turn reasoning off, such as GPT-6 Astra, cannot call tools on Chat Completions at all. To keep reasoning on tool turns, serve the route through `/v1/responses` (see [Protocols and prompt caching](#protocols-and-prompt-caching)).

Kimi K2.5/K2.6 use an on/off thinking switch; K2.7 Code/Highspeed is always thinking. Kimi K3 and DeepSeek V4 use public effort evidence with LiteLLM overrides. Responses levels are translated to valid `reasoning.effort` values; Chat-only compatibility fields never reach Responses models. Legacy cached level maps remain usable while discovery refreshes, and stored Moonshot compatibility can restore safe display normalization.

The development probe runs against minimized snapshots with `npm run probe:proxy -- --snapshot tests/fixtures/proxy/prod-model-info-2026-09-04.json`. A bounded live matrix is available with `--live --base-url <proxy-url> --max-requests 100`; its request bodies use the discovered carrier and request policy.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| No models, or no new routes, while `PI_OFFLINE` is set | Pi skips every model network refresh, including the one `/model` starts. Run `/litellm-refresh`, or unset `PI_OFFLINE` (`PI_OFFLINE=0` is not enough) |
| `No models available` at startup, gone after a restart | A Pi startup race, not discovery — see [`No models available` at startup](#no-models-available-at-startup) |
| "discovered no models" | Proxy returned an empty list — check pi's startup log and verify `/model/info`, `/v1/models`, or `/health` responds |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models. Offline mode does not recover a root mismatch, including a different path prefix — see [Model host enforcement](#model-host-enforcement) |
| A provider shows no models | The base URL is missing, invalid, still the placeholder, or its full root, including any path prefix, differs from the root in the cached catalog. Check stderr and see [Model host enforcement](#model-host-enforcement) |
| A configured or restored model fails to complete but never leaks the credential | Its `api` is absent from the current provider catalog, so Pi used global API fallback with the credential root pinned. Open `/model` against the active proxy to repopulate the protocol — see [Model host enforcement](#model-host-enforcement) |
| `LiteLLM discovery: ... route group(s) have missing or conflicting deployment provider evidence` | One or more deployments lack resolvable backend authority or resolve to different providers or concrete catalog models. Add consistent `litellm_params.model`, `model_info.base_model`, or adapter metadata; catalog-derived limits, pricing, and reasoning metadata are withheld meanwhile. This may be intentional for cross-host routes such as Vertex/Bedrock Claude. |
| `LiteLLM discovery: ... route group(s) mix chat-style and explicitly incompatible deployment modes` | A public route can select both a chat-style deployment and an incompatible deployment such as an embedding model. Give the embedding deployment its own `model_name`. |
| `LiteLLM (...): the request for "<route>" was answered by a fallback` | A successful HTTP response reports a LiteLLM router fallback (`x-litellm-attempted-fallbacks`). `/model/info` does not expose fallbacks, so protocol and model handling follow the requested route's own deployments. If the primary and fallback groups support Chat Completions, set `model_info.supported_endpoints: ["/v1/chat/completions"]` on the primary to avoid Responses-to-Chat bridging. This warning does not change routing or fix upstream bridging defects. It names only the requested route; LiteLLM reports the group and model that answered in the `x-litellm-model-group` and `x-litellm-model-name` response headers. Warned once per provider and route until the extension reloads; shown in the UI, or stderr without a UI. |
| A model is marked ` (incomplete metadata)` | `/model/info` or `/health` did not provide enough authoritative metadata. Explicit fields remain usable, but unknown cost fields are shown as zero and route-name cache enrichment stays disabled. |
| `401 Token expired` | Set `LITELLM_API_KEY_HELPER`. |
| No models with gcloud auth | Verify `gcloud auth application-default login` has been run or set `GOOGLE_APPLICATION_CREDENTIALS` to an `authorized_user` ADC file |
| Enterprise SSO waits for token insertion | The proxy returned 404/405 for `/sso/cli/start`, so Pi used the legacy flow — upgrade LiteLLM or paste the UI token |
| Enterprise CLI SSO start/poll fails | Check the proxy logs and verify `/sso/cli/start` and `/sso/cli/poll/{login_id}` are reachable; only 404/405 falls back to legacy login |
| Enterprise SSO login shows "virtual key generation failed" | The LiteLLM instance may lack a database (`/key/generate` requires one), your user account may lack key-generation permission, or the request timed out; the JWT is used directly as a fallback |
| Enterprise SSO token prompt fails with "SSO token is required" | The token field was left empty — paste the token copied from the LiteLLM UI |
| MCP tools not showing | Run `/mcp` and check the provider's server: its state shows a connection or sign-in error. Verify the proxy serves `/mcp` and that the key has MCP access |
| MCP tools listed but the model cannot call them | `codemode` and `deferred` tools are reached through Pi's `codemode` and `tool_search` tools. Check that neither is disabled; `--no-extensions` disables them and Pi's MCP client alike. Setting `litellm.mcp.exposure` to `direct` declares the tools on every request instead |
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
