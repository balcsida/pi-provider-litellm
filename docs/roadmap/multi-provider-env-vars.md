---
status: committed
done-when: Users can define and authenticate multiple LiteLLM providers entirely through environment variables (via LITELLM_PROVIDER_<NAME>_*, LITELLM_PROVIDERS, or LITELLM_PROVIDERS_JSON) and run /login <alias>, with full test coverage and green npm run check.
---

# Multi-Provider Configuration via Environment Variables Implementation Plan

**Goal:** Enable zero-config multi-proxy LiteLLM setups in Pi via dynamic environment variable scanning (`LITELLM_PROVIDER_<NAME>_*`, `LITELLM_PROVIDERS`, `LITELLM_PROVIDERS_JSON`) with full authentication feature parity across all providers.
**Architecture:** `getProviderDefinitions` parses disk settings, JSON env, canonical lists, and prefix-scanned environment variables into `ProviderDefinition` records. Each definition carries its scoped `envPrefix`, enabling per-provider resolution for base URLs, API keys, command helpers, Google ADC, and OAuth without leaking global `LITELLM_*` variables to secondary providers.
**Tech Stack:** TypeScript, Node.js (>=22.19.0), Vitest, `@earendil-works/pi-ai`, `@earendil-works/pi-coding-agent`.

---

### Task 1: Environment Variable Parsing & Multi-Provider Discovery (`getProviderDefinitions`)

**Context:**
Currently, `getProviderDefinitions` only loads secondary providers declared statically in `settings.json` under `litellm.providers`. This task adds dynamic discovery of secondary providers from environment variables: prefix-scanned variables (`LITELLM_PROVIDER_<NAME>_*`), an optional canonical provider list (`LITELLM_PROVIDERS`), and a structured JSON string (`LITELLM_PROVIDERS_JSON`). It also enforces provider ID validation and merges environment-defined providers with `settings.json`.

**Files:**
- Modify: `src/index.ts`
- Create: `tests/multi-provider-env.test.ts`

**What to implement:**
1. Update `ProviderDefinition` type in `src/index.ts`:
   ```ts
   type ProviderDefinition = {
     name: string;
     displayName: string;
     baseUrl?: string;
     apiKeyConfig?: string;
     headers?: unknown;
     envPrefix?: string;
     useDefaultEnv: boolean;
     useGcloudTokenAuth: boolean;
     enableOAuth: boolean;
     allowInsecureHttp: boolean;
     oidc?: unknown;
   };
   ```
2. Implement parsing and scanning helpers in `src/index.ts`:
   - `parseProviderEnvVars(env: NodeJS.ProcessEnv)`:
     - Scans `Object.keys(env)` for `^LITELLM_PROVIDER_([A-Z0-9_]+)_(BASE_URL|API_KEY|API_KEY_HELPER|HEADERS|DISPLAY_NAME|NAME|ALLOW_INSECURE_HTTP|USE_GCLOUD_AUTH|ENABLE_OAUTH)$`.
     - Extracts the uppercase token (e.g. `CORP_EAST`) and maps to property keys.
   - `parseCanonicalProviderList(raw: string | undefined)`:
     - Splits comma/whitespace-separated string into provider ID strings (e.g. `["corp-east", "dev"]`).
   - `parseProvidersJson(raw: string | undefined)`:
     - Safely parses JSON object and normalizes raw provider settings.
   - `isValidProviderName(name: string)`:
     - Validates against `^[a-z0-9][a-z0-9_-]{0,63}$` and rejects reserved tool/command names (`mcp`, `skills`, `codemode`, `tool_search`, `litellm`).
3. Update `getProviderDefinitions(settings)` in `src/index.ts`:
   - Keep primary `"litellm"` provider with `useDefaultEnv: true`.
   - Incorporate `LITELLM_PROVIDERS_JSON`.
   - Incorporate `LITELLM_PROVIDERS` and prefix-scanned `LITELLM_PROVIDER_<NAME>_*` variables.
   - For scanned unlisted tokens, derive default provider ID via `token.toLowerCase().replace(/_/g, "-")` unless overridden by `_NAME`.
   - Merge with `settings.json` (disk settings override env vars for matching provider fields; non-overlapping providers combine additively; `enabled === false` excludes the provider).

**Steps:**
- [ ] Write failing unit tests in `tests/multi-provider-env.test.ts` verifying:
  - Discovery of single and multiple providers via `LITELLM_PROVIDER_<NAME>_BASE_URL` and `_API_KEY`.
  - Default token normalization (`TEAM_A` $\rightarrow$ `team-a`).
  - Override via `LITELLM_PROVIDER_<NAME>_NAME`.
  - Canonical list preservation via `LITELLM_PROVIDERS="team-a,team_b"`.
  - JSON provider parsing via `LITELLM_PROVIDERS_JSON`.
  - Precedence: `settings.json` overriding env vars; `enabled: false` exclusion.
  - Rejection of invalid or reserved provider names.
- [ ] Run `npm test -- tests/multi-provider-env.test.ts`
  - Ensure tests fail with expected compilation/assertion errors.
- [ ] Implement provider env scanning and `getProviderDefinitions` in `src/index.ts`.
- [ ] Run `npm test -- tests/multi-provider-env.test.ts`
  - Ensure all new unit tests pass.
- [ ] Run `npm run check`
  - Verify formatting, linting, typecheck, and supply-chain guards pass.
- [ ] Commit with message: `feat: discover multi-provider definitions from environment variables`

**Acceptance criteria:**
- [ ] Secondary providers defined via `LITELLM_PROVIDER_<NAME>_*` are populated in `getProviderDefinitions`.
- [ ] `LITELLM_PROVIDERS` and `LITELLM_PROVIDERS_JSON` are supported and properly merged.
- [ ] Invalid and reserved provider names are safely filtered out.
- [ ] Existing `settings.json` behavior remains 100% backward compatible.

---

### Task 2: Per-Provider Credential & Endpoint Resolution

**Context:**
Secondary providers must resolve their configuration and credentials from their scoped environment variables (`LITELLM_PROVIDER_<NAME>_*`) without leaking or falling back to the global `LITELLM_*` variables (which belong solely to the primary `"litellm"` provider). Furthermore, secondary providers must support Google ADC (when `useGcloudTokenAuth: true`) and PKCE OAuth (when `enableOAuth: true`).

**Files:**
- Modify: `src/index.ts`
- Modify: `tests/multi-provider-env.test.ts`
- Modify: `tests/index.test.ts`

**What to implement:**
1. Update `configuredBaseUrl` in `src/index.ts`:
   - Checks `credential.env.LITELLM_BASE_URL` $\rightarrow$ `definition.baseUrl` $\rightarrow$ `definition.envPrefix ? ctx.env(`${definition.envPrefix}_BASE_URL`)` $\rightarrow$ (if `definition.useDefaultEnv`) `ctx.env("LITELLM_BASE_URL")`.
2. Update `resolveApiKeyAuth` and `resolveCredentials` in `src/index.ts`:
   - Checks stored key $\rightarrow$ Google ADC (if `definition.useGcloudTokenAuth`) $\rightarrow$ `definition.apiKeyConfig` $\rightarrow$ `definition.envPrefix ? ctx.env(`${definition.envPrefix}_API_KEY_HELPER`)` $\rightarrow$ `definition.envPrefix ? ctx.env(`${definition.envPrefix}_API_KEY`)` $\rightarrow$ (if `definition.useDefaultEnv`) `LITELLM_API_KEY_HELPER` $\rightarrow$ `LITELLM_API_KEY`.
3. Update `apiKey.check` in `src/index.ts`:
   - Inspects `definition.envPrefix` to check whether scoped credentials or helpers are available and reports the appropriate source name.
4. Update `resolveHeaders` and `resolveHeadersFromContext` in `src/index.ts`:
   - Checks `definition.headers` $\rightarrow$ `definition.envPrefix ? env(`${definition.envPrefix}_HEADERS`)` $\rightarrow$ (if `definition.useDefaultEnv`) `env("LITELLM_HEADERS")`.
5. Maintain strict host pinning via `requireCredentialRoot` across all providers.

**Steps:**
- [ ] Write failing tests in `tests/multi-provider-env.test.ts` verifying:
  - Scoped `_BASE_URL` and `_API_KEY` are used for secondary providers.
  - Global `LITELLM_BASE_URL` and `LITELLM_API_KEY` are NOT used by secondary providers.
  - Scoped `_API_KEY_HELPER` command is lazily executed for secondary providers.
  - Scoped `_HEADERS` JSON is resolved and passed.
  - Scoped `_USE_GCLOUD_AUTH="1"` enables Google ADC resolution.
  - Scoped `_ENABLE_OAUTH="1"` enables OAuth descriptor on provider auth.
- [ ] Run `npm test -- tests/multi-provider-env.test.ts`
  - Ensure tests fail with expected assertions.
- [ ] Implement scoped credential, endpoint, and header resolution in `src/index.ts`.
- [ ] Run `npm test -- tests/multi-provider-env.test.ts`
  - Ensure tests pass.
- [ ] Run `npm test`
  - Ensure all existing test suites pass.
- [ ] Commit with message: `feat: resolve per-provider credentials and headers for env providers`

**Acceptance criteria:**
- [ ] Secondary providers resolve their own scoped `LITELLM_PROVIDER_<NAME>_*` variables.
- [ ] Primary provider global variables do not leak into secondary providers.
- [ ] Command helpers, Google ADC, and OAuth work cleanly on secondary providers.

---

### Task 3: Universal Interactive CLI Login (`/login <alias>`) for Secondary Providers

**Context:**
Currently, `apiKey.login` is only defined for `definition.name === PROVIDER_NAME` (`"litellm"`), making interactive `/login <alias>` unavailable for secondary providers. This task enables interactive login for all registered providers, allowing users to run `/login <alias>` to enter or update endpoints and credentials interactively.

**Files:**
- Modify: `src/index.ts`
- Modify: `tests/multi-provider-env.test.ts`

**What to implement:**
1. Update `createProviderAuth` in `src/index.ts`:
   - Provide `apiKey.login` for all providers:
     ```ts
     login: async (interaction: AuthInteraction) => {
       loginHooks?.start();
       const result = await loginApiKey(interaction, definition);
       return completeLogin(result);
     }
     ```
   - Provide `oauth.login` when `definition.enableOAuth` is `true`.
2. Update `knownBaseUrl(definition)` in `src/index.ts`:
   - Inspects `definition.baseUrl` $\rightarrow$ `definition.envPrefix ? process.env[`${definition.envPrefix}_BASE_URL`] : undefined` $\rightarrow$ (if `definition.useDefaultEnv`) `process.env.LITELLM_BASE_URL` $\rightarrow$ stored `auth.json` URL.
3. Wire `loginHooks` per definition so that `/login <alias>` triggers `dropMcpServer(definition)` on start.

**Steps:**
- [ ] Write failing test in `tests/multi-provider-env.test.ts` verifying:
  - `apiKey.login` is defined for secondary providers.
  - `knownBaseUrl` offers the provider-scoped base URL for `/login <alias>`.
  - Logging in as `/login <alias>` saves credentials to `auth.json[alias]` and drops existing MCP connections.
- [ ] Run `npm test -- tests/multi-provider-env.test.ts`
  - Verify test failure.
- [ ] Update `createProviderAuth` and `knownBaseUrl` in `src/index.ts`.
- [ ] Run `npm test -- tests/multi-provider-env.test.ts`
  - Verify test passes.
- [ ] Run `npm test`
  - Verify full test suite passes.
- [ ] Commit with message: `feat: enable interactive /login for secondary providers`

**Acceptance criteria:**
- [ ] `/login <alias>` functions interactively for any secondary provider.
- [ ] Known base URLs are populated from scoped env vars or past logins.
- [ ] MCP server is dropped and re-synced upon login completion.

---

### Task 4: Documentation & Final Verification

**Context:**
Users need clear documentation in `README.md` on how to configure multiple LiteLLM providers via environment variables, explaining the `LITELLM_PROVIDER_<NAME>_*` syntax, `LITELLM_PROVIDERS` list, `LITELLM_PROVIDERS_JSON`, and `/login <alias>`.

**Files:**
- Modify: `README.md`

**What to implement:**
- Add a dedicated section in `README.md` under Configuration describing multi-provider environment variables.
- Include concrete examples for Kubernetes secrets, Docker `-e`, `.env` files, and JSON injection.
- Document precedence rules between `settings.json`, environment variables, and `auth.json`.

**Steps:**
- [ ] Update `README.md` with multi-provider environment variable documentation.
- [ ] Run `npm run check` (biome, typecheck, vitest, supply-chain guard).
- [ ] Run `npm pack --dry-run` to ensure package bundle integrity.
- [ ] Commit with message: `docs: document multi-provider environment variable configuration`

**Acceptance criteria:**
- [ ] `README.md` comprehensively documents all multi-provider environment variable patterns.
- [ ] `npm run check` passes with zero warnings or errors.
- [ ] Package manifest and supply-chain guards remain intact.
