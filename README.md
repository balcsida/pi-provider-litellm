# pi-provider-litellm

LiteLLM proxy native Provider extension for [Pi](https://pi.dev). Pi 0.83.0+ is required.

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

If your LiteLLM proxy supports SSO/OAuth authentication, Pi selects its supported browser login flow automatically:

1. Run `/login litellm` inside pi and select `Sign in with LiteLLM SSO`
2. Confirm the offered proxy URL, or enter one if this is the first login
3. Complete sign-in in the browser Pi opens, then return to Pi

Pi first checks `/.well-known/litellm-cli-auth`. A proxy exposing the supported CLI-auth contract uses authorization code + PKCE (`S256`) with a temporary `127.0.0.1` callback; the browser must run on the same machine as Pi. Advertised endpoints must stay on the proxy's origin, and discovery, registration, and token requests do not follow HTTP redirects. Invalid discovery stops login.

Pi stores PKCE access and refresh tokens in `~/.pi/agent/auth.json` with file mode `0600`, refreshes automatically, and saves each rotated token pair. Temporary network failures or HTTP 429/5xx responses allow reuse of the existing access token only until its exact expiry. Rejected or revoked refresh credentials require `/login litellm` again.

If discovery returns 404, Pi uses `/sso/cli/start`: confirm the verification code in the browser, then select a team in Pi if prompted. Pi polls for the short-lived CLI token and uses its advertised lifetime, falling back to `LITELLM_CLI_JWT_EXPIRATION_HOURS` or LiteLLM's 24-hour default on older proxies. Only LiteLLM versions that gate this older CLI SSO flow need `EXPERIMENTAL_UI_LOGIN=True` on the proxy.

If `/sso/cli/start` returns 404 or 405, copy the token from the LiteLLM UI and paste it into Pi, with an optional virtual-key exchange. Pi reads JWT expiry claims and prompts for login when re-authentication is required.

Tokens are stored in Pi's local file, not an OS keychain. `/logout litellm` deletes the local credential only; logging out or signing in again does not revoke the previous session on the proxy because Pi has no provider revocation hook.

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
| `baseUrl` | `LITELLM_BASE_URL` for `litellm`; required for aliases | LiteLLM proxy URL, with or without `/v1`. Must be a full `http`/`https` URL; a provider with no resolvable base URL exposes no models (see [Model host enforcement](#model-host-enforcement)) |
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

To refresh catalogs on demand, run `/litellm-refresh` for every configured LiteLLM provider, or `/litellm-refresh <provider>` for one alias. It works with `PI_OFFLINE` set, which stops Pi's own `/model` refresh from contacting the network. It contacts only the configured proxies: `LITELLM_OFFLINE=1` and `LITELLM_DISCOVERY_TIMEOUT_MS=0` still disable it, and models.dev enrichment stays off under `PI_OFFLINE`. Each provider reports its model count, the failure, or missing credentials. With LiteLLM MCP enabled, the refresh also re-checks that provider's MCP tools.

Every request routed through a configured LiteLLM provider includes Pi's canonical session ID in the
`x-litellm-session-id` header. This groups Chat Completions, Responses, and native Messages requests in LiteLLM without
adding transport-specific fields to request bodies.

## Model transport

A `/model/info` route group uses native Anthropic `/v1/messages` only when every deployment explicitly reports Chat mode, identifies a Claude backend through an Anthropic-, Bedrock-, or Vertex-capable adapter, and every declared backend identity resolves the same Anthropic compatibility policy. A `claude-<name>-<major>-<minor>` backend id resolves through its `claude-<name>-<major>` entry when the exact id is absent; an unknown generation stays on Chat Completions. When `supported_endpoints` is present, it must include `/v1/messages`; an explicit endpoint list without it keeps the group on Chat Completions. An unresolved routing or base-model identity denies native Messages rather than being discarded from the unanimity check. Mixed-generation Claude groups can use Messages when their serializer policies match, with reasoning levels restricted to their common supported set. Groups with incompatible policies, mixed families, or unknown backend evidence remain on Chat Completions; unanimous explicit Responses mode takes precedence. Routes discovered through `/health` never select native Messages: Messages candidates are downgraded to Chat Completions with reasoning controls closed, while correlated Responses detail remains on Responses. Evidence-free `/v1/models` and wholly health-only `/health` groups take transport and presentation metadata from the bounded Pi catalog lookup: `openai-responses` when the catalog says so, otherwise Chat Completions. They expose no speculative reasoning selector and never select native Messages. Catalog identity such as Amazon Bedrock remains separate from the selected wire protocol.

Native Messages authenticates with `x-api-key`; every transport carries the `x-litellm-session-id` header described above. Strict JSON-schema tools are enabled separately from Messages transport and thinking compatibility: every deployment must affirm support for the actual route/API. Generic `supports_response_schema` and `supports_native_structured_output` metadata does not grant strict tools. Only deployments that LiteLLM reports with the `anthropic` adapter carry that evidence today. Claude served through Bedrock or Vertex therefore uses ordinary tool serialization, and a tool that requires strict sampling fails before the request is sent.

## Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `LITELLM_API_KEY_HELPER` | unset | Command that prints a fresh LiteLLM bearer token. Takes precedence over `LITELLM_API_KEY`. The extension runs it while resolving request auth, and Pi's per-request auth path is uncached, so rotating/short-lived tokens stay fresh. |
| `LITELLM_HEADERS` | unset | JSON object of extra headers sent to LiteLLM provider, discovery, MCP, and Skills Gateway requests. Provider aliases can use it with `"headers": "$LITELLM_HEADERS"`. |
| `LITELLM_GCLOUD_TOKEN_AUTH` | unset | If set to a non-empty value other than `0`, use Google Application Default Credentials as the LiteLLM bearer token source. This takes precedence over `LITELLM_API_KEY_HELPER` and `LITELLM_API_KEY` when no stored `/login litellm` credential exists. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google default ADC path | Optional path to an ADC JSON file used by `LITELLM_GCLOUD_TOKEN_AUTH`. If unset, the extension checks the default gcloud ADC locations. |
| `LITELLM_OFFLINE` | unset | If `1`, disable all model and MCP discovery, including post-login discovery; use cached models only when their stored canonical proxy root exactly matches the active credential root, including any path prefix. URL-standard host casing and default ports are canonicalized, but paths remain case-sensitive. |
| `PI_OFFLINE` | unset | Pi documents it as disabling automatic network activity, including model catalog refreshes. Through Pi 0.87.1, any set value (including `0` or an empty string) disables startup model discovery and `/model` network refreshes. Run `/litellm-refresh`, or unset it, to discover; cached models can still be used offline. |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Background and explicit discovery fetch timeout in ms; `0` disables automatic discovery |
| `LITELLM_CLI_JWT_EXPIRATION_HOURS` | `24` | CLI SSO token lifetime fallback for older proxies whose poll response omits `expires_in`; mirror a non-default proxy setting locally |
| `LITELLM_VERBOSE_DISCOVERY` | unset | If `1`, enable progress messages during model and MCP discovery (login, refresh, startup), including startup skip reasons, defaulted `/model/info` context windows, and MCP prepared/registered/dropped counts. Progress messages are off by default; MCP safety diagnostics (see below) are always reported regardless of this setting |
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

If your LiteLLM proxy exposes MCP REST endpoints, this extension discovers tools from:

- `GET /mcp-rest/tools/list`
- `POST /mcp-rest/tools/call`

An MCP access denial pauses discovery until a successful `/login litellm` (SSO or API key); an alias provider, which has no login, stays paused until its credentials, base URL, or headers change. The pause survives restarts in `~/.pi/agent/litellm-mcp-pauses/`, scoped to the proxy, login, and headers with salted credential fingerprints. A new API key or header configuration has its own scope. Each explicit login creates a fresh opaque `litellmMcpSession` identifier in Pi's stored credential, even when the key is unchanged; automatic OAuth refresh preserves that identifier and its pause. Older Pi processes cannot pause the new login. Pause files contain no credentials or proxy error text. Model discovery and chat continue normally. Grant MCP access on the proxy before signing in again. An explicit `litellm.mcp.enabled: false` setting remains disabled after login.

Each discovered tool is registered as a native Pi tool named `mcp_<server>_<tool>_<hash>`, at most 64 characters, using only `[a-z0-9_]`. The trailing 10-character hash is derived from the tool's identity (`server_id`, `server_name`, `name`) and is always present, so a name depends only on that tool and never on which other tools happen to be in the same catalog — adding or removing a sibling never renames a survivor. When the readable prefix would overflow 64 characters it is truncated from the right, so a long server name can leave little or none of the tool name visible; the hash is what distinguishes such tools. Tools from different servers therefore never overwrite each other, and exact duplicate identities are registered once. Each alias provider discovers its own catalog: its tools are named `mcp_<alias>_<server>_<tool>_<hash>`, with the alias name also folded into the hash, and are called with that alias's credentials, so two providers exposing the same server never replace each other's tools. The default `litellm` provider keeps unprefixed names, and diagnostics for an alias read `LiteLLM MCP ("<alias>"):`.

MCP discovery runs once for every configured provider before the first agent turn (so `-p` and other non-interactive modes get MCP tools), after Pi refreshes LiteLLM models, and after `/login litellm`; extension activation never waits for it. Discovery accepts at most a 5 MiB response and registers at most 512 tools. The response must be a JSON array of tools or an object with a `tools` array; an `unexpected_error` or unknown non-null `error` in the object envelope, an object without `tools`, and every other shape are reported as discovery failures rather than as empty catalogs. Failure reports use extension-owned text and never include LiteLLM's proxy-supplied tag or `message`. A `partial_failure` envelope keeps and registers the returned tools while reporting one bounded generated diagnostic. Each tool is isolated during normalization, so one bad tool never hides valid siblings. A tool is accepted only when its schema has a root of exactly `"type": "object"` (a bare `properties` object with no `type`, or a type union such as `["object","null"]`, is refused), plain-object `properties`, at most 16 levels of JSON nesting (about eight nested schema objects when each level uses `properties`), and a serialized size of at most 64 KiB. Descriptions are limited to 4 KiB, and tool labels and result metadata to 256 bytes, each with a truncation marker.

### Which schema a tool ends up with

A tool registers with one of two parameter shapes.

- **The extension-owned `args` envelope** — a single object-valued `args` property, required at execution. Used when the proxy supplied no schema at all (absent or `{}`), and used as a safe substitute when the supplied schema could not be proven safe (below). Nothing in this shape originates from the proxy.
- **The supplied schema, passed through unchanged.** Pi validates arguments against it before the call and may first coerce supported primitive values: omitting a property listed in `required` is rejected, while a value of the wrong type is rejected only when it cannot be coerced to the declared type. Unknown extra properties are accepted by default, but an `additionalProperties` schema can validate and coerce them, and constraints such as `additionalProperties: false` or `unevaluatedProperties: false` can reject them. Passing through is not the same as skipping validation. A schema of exactly `{"type": "object"}` with no `properties` therefore accepts any arguments at the top level rather than nesting them under `args`. Real schema properties named `args`, `properties`, or `required` remain intact.

An `inputSchema` that is present but is not a JSON object (a string, number, or array) is a malformed tool and is dropped as `invalid-schema`. It is not treated as schemaless, so a proxy emitting junk cannot obtain the permissive envelope.

### Regexes and references in supplied schemas

`pattern`, and the keys of `patternProperties`, are compiled into a backtracking regular expression and executed with no time limit, so a proxy-supplied expression can block Pi for minutes on a single tool call. A `$ref` resolves an arbitrary JSON pointer, which means an expression can be parked under any key at all, including data-only positions such as `default` or `examples`. Enumerating "positions where a subschema may appear" is therefore not a sound defence.

Instead, the whole supplied document is walked — every key and every value — and the tool keeps its schema only if that walk completes and finds nothing dangerous. A walk stops short, and the tool is degraded, when it finds a `pattern` or `patternProperties` keyword at all — whatever its value type, since refusing to vouch for it does not depend on an upstream type guard staying as it is — a `$dynamicRef` or `$recursiveRef`, a `$ref` pointing outside the document, or a graph too deep or too large to inspect within budget. A local `#/...` pointer is accepted only when the walk covers its target *and* that target resolves to something usable as a subschema, with the reference chain proven acyclic — covering the document proves no expression hides behind a pointer, but not that the pointer names a schema at all. A pointer to a non-schema (`#/description`), a pointer with no target (`#/$defs/missing`), an `$anchor` reference (`#name`), and a mutual `$defs` cycle are each degraded rather than forwarded, because the validator would otherwise throw, exhaust the stack, or produce a tool that can never accept any argument. A key that is an argument *name* rather than a keyword — a property literally called `pattern` or `patternProperties` — is left alone.

**Degrading replaces the schema, not the tool.** Because regexes are common in real MCP schemas, a tool in this position still registers, with the `args` envelope in place of its schema, and is reported under the `schema-envelope` class. The compatibility cost is real and worth knowing: such a tool loses validation against the proxy-supplied schema but retains Pi's validation of the extension-owned envelope, so its arguments must be passed as an object under `args` and the model sees a less specific contract than the server documents. The server still validates authoritatively, so an argument that violates the proxy-specific contract becomes a server-side tool error rather than a local one.

### What you see when a tool is missing or degraded

Every raw entry the proxy returns is accounted for exactly once, and the counts reconcile: `raw = prepared + dropped + beyond-the-cap`, where *prepared* is what will be registered. If a registration pass is refused partway, the shortfall between prepared and registered is reported separately by that pass's own diagnostic. A tool is **dropped** for one of four reasons — `invalid-tool` (no usable name or server identity), `duplicate-identity`, `invalid-schema`, or `name-collision` — or discarded for being beyond the 512-tool limit (`tool-cap`). A tool is **degraded** rather than dropped under `schema-envelope`. Each class is reported with a count and a bounded sample of names through Pi notifications when a UI is active, or stderr in non-interactive mode. MCP messages emitted before the TUI context is ready are buffered until session start. For example:

```text
LiteLLM MCP: kept 2 MCP tools but replaced their schema with a safe args envelope, because of a schema that could not be proven free of proxy-supplied regexes or references: mcp_srv_lookup_1c97861c1f, mcp_srv_matcher_3ee4c31a5a.
LiteLLM MCP: dropped 1 MCP tool with a missing name or server identity: entry_3.
```

A class is re-reported whenever its membership changes, including a change beyond the names shown in the sample, and stays quiet while unchanged — so a stable proxy does not repeat itself across refreshes, while a changing one is always surfaced. A class that stops occurring is cleared, so its recurrence is reported rather than suppressed. These lines never contain the proxy's schema, its response body, its descriptions, or your API key; only generated names, positional placeholders, and counts. They are reported regardless of `LITELLM_VERBOSE_DISCOVERY`; set that variable to also see per-refresh raw, prepared, enveloped, and registered counts. The enveloped count covers every tool carrying the envelope, whether because the proxy supplied no schema or because its schema was degraded, so it is the number of tools with no proxy-specific argument contract.

### Registration lifecycle

Pi registers a tool synchronously, replacing any existing tool of the same name, and exposes no way to unregister. It has no notion of rejecting one tool while accepting another: the only failure is a staleness check that fires once the extension instance has been superseded, and that check is never reset. A refused registration therefore ends the whole pass rather than skipping one tool.

So a pass that is refused partway leaves the tools registered up to that point, reports one bounded diagnostic, and makes no further attempt from that extension instance — retrying would re-run discovery on every refresh and could never succeed. A reload creates a fresh instance, which starts clean on its own. Because names are stable and registration replaces by name, any retry that does happen is idempotent.

Network discovery is separate and remains retryable: a catalog that yields no registrable tool is not recorded as settled, so a later refresh tries again, and the empty result is reported rather than passing in silence. A tool that disappears from the proxy's catalog stays registered until Pi restarts.

MCP tools run in Pi's parallel tool mode. Each side-effecting `POST /mcp-rest/tools/call` is attempted exactly once: timeouts, connection failures, HTTP errors, and malformed responses are returned to Pi as tool errors rather than retried. Pi cancellation aborts an in-flight call and preserves its original cancellation reason. Tool-call response bodies are limited to 5 MiB before JSON parsing, and returned result or error text to 64 KiB with a truncation marker.

A passed-through schema's `format` keyword is evaluated, using the validator's own built-in expressions rather than anything the proxy supplies.

Tools discovered this way are only as trustworthy as the MCP servers behind your proxy: their descriptions and results are text the model reads.

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

Discovery selects native Messages for proven Claude groups under the [Model transport](#model-transport) rules, Responses for OpenAI-family backends, and Chat Completions for other OpenAI-compatible backends. Azure and Azure AI deployments stay on Chat Completions unless an explicit Responses mode or a `supported_endpoints` list containing `/v1/responses` authorizes Responses; `api_version` alone does not prove support. Azure is detected from the adapter field, an `azure/` or `azure_ai/` model prefix, or the reported provider; other adapters follow the backend family because LiteLLM bridges `/v1/responses` to Chat Completions when the provider has no native Responses config (`litellm/responses/main.py`, `_bridges_to_chat_completions`). A route whose model prefix and `custom_llm_provider` name different providers is treated as unidentified and stays on Chat Completions, so it also does not get the local 128-tool preflight. When `/model/info` supplies `supported_endpoints`, that list takes precedence, and `mode: "responses"` remains an explicit Responses signal. An evidence-free fallback entry—a bare `/v1/models` id or a `/health` entry with no detail row—takes its protocol from the Pi catalog entry for its id; both evidence-free fallback paths also inherit presentation metadata from the bounded catalog lookup. An unknown id stays on Chat Completions. The route name alone authorizes nothing. A concrete id expanded from wildcard `/model/info` routes inherits the deployment evidence of the wildcard route LiteLLM would select for it, with the requested id substituted into the row's `model` the way LiteLLM serves it; the id is omitted when that selected route is not a published chat-style route. Chat Completions and Responses use `<root>/v1`; native Messages uses the proxy root directly. All three protocols pass through the provider's host guard. When an evidence-free cached `(no metadata)` entry can be enriched from the Pi catalog, it is restored on Responses only for a catalog Responses model and otherwise on Chat Completions, never Anthropic Messages. A model supplied only through `models.json` whose `api` this provider does not implement takes Pi's global API fallback rather than this provider's host guard; the credential-root pin (see [Model host enforcement](#model-host-enforcement)) still keeps the credential on the active proxy.

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
| `LiteLLM (...): a fallback served "<route>"` | A successful HTTP response reports a LiteLLM router fallback (`x-litellm-attempted-fallbacks`). `/model/info` does not expose fallbacks, so protocol and model handling follow the requested route's own deployments. If the primary and fallback groups support Chat Completions, set `model_info.supported_endpoints: ["/v1/chat/completions"]` on the primary to avoid Responses-to-Chat bridging. This warning does not change routing or fix upstream bridging defects. Warned once per provider and route until the extension reloads; shown in the UI, or stderr without a UI. |
| A model is marked ` (incomplete metadata)` | `/model/info` or `/health` did not provide enough authoritative metadata. Explicit fields remain usable, but unknown cost fields are shown as zero and route-name cache enrichment stays disabled. |
| `401 Token expired` | Set `LITELLM_API_KEY_HELPER`. |
| No models with gcloud auth | Verify `gcloud auth application-default login` has been run or set `GOOGLE_APPLICATION_CREDENTIALS` to an `authorized_user` ADC file |
| Enterprise SSO waits for token insertion | The proxy returned 404/405 for `/sso/cli/start`, so Pi used the legacy flow — upgrade LiteLLM or paste the UI token |
| Enterprise CLI SSO start/poll fails | Check the proxy logs and verify `/sso/cli/start` and `/sso/cli/poll/{login_id}` are reachable; only 404/405 falls back to legacy login |
| Enterprise SSO login shows "virtual key generation failed" | The LiteLLM instance may lack a database (`/key/generate` requires one), your user account may lack key-generation permission, or the request timed out; the JWT is used directly as a fallback |
| Enterprise SSO token prompt fails with "SSO token is required" | The token field was left empty — paste the token copied from the LiteLLM UI |
| MCP tools not showing | Verify the proxy exposes `/mcp-rest/tools/list` and open `/model` after fixing the proxy |
| Some MCP tools missing or unvalidated | Check Pi notifications (stderr in non-interactive mode) for `LiteLLM MCP:` messages. Dropped tools are reported under `invalid-tool`, `duplicate-identity`, `invalid-schema`, `name-collision`, or `tool-cap`, each with a count and a bounded sample of generated names. A tool reported under `schema-envelope` is still present but uses the `args` envelope instead of its own schema. A refused registration pass is reported once. Set `LITELLM_VERBOSE_DISCOVERY=1` for per-refresh raw/prepared/enveloped/registered counts |
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
