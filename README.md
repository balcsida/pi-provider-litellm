# pi-provider-litellm

LiteLLM proxy native Provider extension for [Pi](https://pi.dev). Pi 0.81.0+ is required; the TypeScript source
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

Stored pi credentials for `litellm` take precedence over `LITELLM_API_KEY`; the environment key is used when no saved credential exists. `LITELLM_BASE_URL` is used when no saved login base URL exists.

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

## Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `LITELLM_API_KEY_HELPER` | unset | Command that prints a fresh LiteLLM bearer token. Takes precedence over `LITELLM_API_KEY`. The extension runs it while resolving request auth, and Pi's per-request auth path is uncached, so rotating/short-lived tokens stay fresh. |
| `LITELLM_HEADERS` | unset | JSON object of extra headers sent to LiteLLM provider, discovery, MCP, and Skills Gateway requests. Provider aliases can use it with `"headers": "$LITELLM_HEADERS"`. |
| `LITELLM_GCLOUD_TOKEN_AUTH` | unset | If set to a non-empty value other than `0`, use Google Application Default Credentials as the LiteLLM bearer token source. This takes precedence over `LITELLM_API_KEY_HELPER` and `LITELLM_API_KEY` when no stored `/login litellm` credential exists. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google default ADC path | Optional path to an ADC JSON file used by `LITELLM_GCLOUD_TOKEN_AUTH`. If unset, the extension checks the default gcloud ADC locations. |
| `LITELLM_OFFLINE` | unset | If `1`, disable all model and MCP discovery, including post-login discovery; use cached models only |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Background and explicit discovery fetch timeout in ms; `0` disables automatic discovery |
| `LITELLM_CLI_JWT_EXPIRATION_HOURS` | `24` | CLI SSO token lifetime fallback for older proxies whose poll response omits `expires_in`; mirror a non-default proxy setting locally |
| `LITELLM_VERBOSE_DISCOVERY` | unset | If `1`, enable progress messages during model and MCP discovery (login, refresh, startup), including MCP prepared/registered/dropped counts. Progress messages are off by default; MCP safety diagnostics (see below) are always reported regardless of this setting |

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

Each discovered tool is registered as a native Pi tool named `mcp_<server>_<tool>_<hash>`, at most 64 characters, using only `[a-z0-9_]`. The trailing 10-character hash is derived from the tool's identity (`server_id`, `server_name`, `name`) and is always present, so a name depends only on that tool and never on which other tools happen to be in the same catalog — adding or removing a sibling never renames a survivor. When the readable prefix would overflow 64 characters it is truncated from the right, so a long server name can leave little or none of the tool name visible; the hash is what distinguishes such tools. Tools from different servers therefore never overwrite each other, and exact duplicate identities are registered once.

MCP discovery runs after Pi refreshes LiteLLM models or after `/login litellm`; extension activation never waits for it. Discovery accepts at most a 5 MiB response and registers at most 512 tools. The response must be a JSON array of tools or an object with a `tools` array; a non-null `error` in the object envelope, an object without `tools`, and every other shape are reported as discovery failures rather than as empty catalogs. The failure report carries LiteLLM's `error` tag, sanitized and bounded to 96 bytes, and never its `message`. Each tool is isolated during normalization, so one bad tool never hides valid siblings. A tool is accepted only when its schema has a root of exactly `"type": "object"` (a bare `properties` object with no `type`, or a type union such as `["object","null"]`, is refused), plain-object `properties`, at most 16 levels of JSON nesting (about eight nested schema objects when each level uses `properties`), and a serialized size of at most 64 KiB. Descriptions are limited to 4 KiB, and tool labels and result metadata to 256 bytes, each with a truncation marker.

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

Every raw entry the proxy returns is accounted for exactly once, and the counts reconcile: `raw = prepared + dropped + beyond-the-cap`, where *prepared* is what will be registered. If a registration pass is refused partway, the shortfall between prepared and registered is reported separately by that pass's own diagnostic. A tool is **dropped** for one of four reasons — `invalid-tool` (no usable name or server identity), `duplicate-identity`, `invalid-schema`, or `name-collision` — or discarded for being beyond the 512-tool limit (`tool-cap`). A tool is **degraded** rather than dropped under `schema-envelope`. Each class is reported to stderr on its own line with a count and a bounded sample of names, for example:

```text
LiteLLM MCP: kept 2 MCP tools but replaced their schema with a safe args envelope, because of a schema that could not be proven free of proxy-supplied regexes or references: mcp_srv_lookup_1c97861c1f, mcp_srv_matcher_3ee4c31a5a.
LiteLLM MCP: dropped 1 MCP tool with a missing name or server identity: entry_3.
```

A class is re-reported whenever its membership changes, including a change beyond the names shown in the sample, and stays quiet while unchanged — so a stable proxy does not repeat itself across refreshes, while a changing one is always surfaced. A class that stops occurring is cleared, so its recurrence is reported rather than suppressed. These lines never contain the proxy's schema, its response body, its descriptions, or your API key; only generated names, positional placeholders, and counts. They are written regardless of `LITELLM_VERBOSE_DISCOVERY`; set that variable to also see per-refresh raw, prepared, enveloped, and registered counts. The enveloped count covers every tool carrying the envelope, whether because the proxy supplied no schema or because its schema was degraded, so it is the number of tools with no proxy-specific argument contract.

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

## Mocked LiteLLM smoke workflow

The `LiteLLM Smoke` GitHub Actions workflow starts VidaiMock and a real LiteLLM proxy on the runner. LiteLLM exposes OpenAI-compatible and Anthropic routes whose upstreams are served by VidaiMock, then this extension's smoke runner discovers those models through LiteLLM and sends `/v1/chat/completions` requests through the proxy.

This keeps the LiteLLM integration path under test but does not call real LLM APIs. No provider API keys or GitHub Models permission are required. The smoke runner also asserts that discovery came from `/model/info` (`LITELLM_SMOKE_EXPECT_SOURCE`) so a silent fallback to `/v1/models` fails the run. The workflow also runs auth checks plus optional Postgres-backed auth checks when `LITELLM_LICENSE` is configured for virtual-key and admin-route behavior, then runs a non-interactive Pi CLI smoke with `--list-models` and `-p` against both the OpenAI-compatible and Anthropic-backed routes, so extension loading, model discovery, and real completion paths are covered without opening the TUI. It also runs an interactive Pi TUI smoke covering `/login litellm` and Pi's native `/model` refresh.

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

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| `No models available` at startup, gone after a restart | A Pi startup race, not discovery — see [`No models available` at startup](#no-models-available-at-startup) |
| "discovered no models" | Proxy returned an empty list — check pi's startup log and verify `/model/info`, `/v1/models`, or `/health` responds |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models |
| `401 Token expired` | Set `LITELLM_API_KEY_HELPER`. |
| No models with gcloud auth | Verify `gcloud auth application-default login` has been run or set `GOOGLE_APPLICATION_CREDENTIALS` to an `authorized_user` ADC file |
| Enterprise SSO waits for token insertion | The proxy returned 404/405 for `/sso/cli/start`, so Pi used the legacy flow — upgrade LiteLLM or paste the UI token |
| Enterprise CLI SSO start/poll fails | Check the proxy logs and verify `/sso/cli/start` and `/sso/cli/poll/{login_id}` are reachable; only 404/405 falls back to legacy login |
| Enterprise SSO login shows "virtual key generation failed" | The LiteLLM instance may lack a database (`/key/generate` requires one), your user account may lack key-generation permission, or the request timed out; the JWT is used directly as a fallback |
| Enterprise SSO token prompt fails with "SSO token is required" | The token field was left empty — paste the token copied from the LiteLLM UI |
| MCP tools not showing | Verify the proxy exposes `/mcp-rest/tools/list` and open `/model` after fixing the proxy |
| Some MCP tools missing or unvalidated | Check pi's stderr for `LiteLLM MCP:` lines. Dropped tools are reported under `invalid-tool`, `duplicate-identity`, `invalid-schema`, `name-collision`, or `tool-cap`, each with a count and a bounded sample of generated names. A tool reported under `schema-envelope` is still present but uses the `args` envelope instead of its own schema. A refused registration pass is reported once. Set `LITELLM_VERBOSE_DISCOVERY=1` for per-refresh raw/prepared/enveloped/registered counts |
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
