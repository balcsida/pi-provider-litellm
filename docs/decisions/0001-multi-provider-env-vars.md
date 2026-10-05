# ADR 0001: Multi-Provider Configuration via Environment Variables

## Status
Accepted

## Context
`pi-provider-litellm` registers a primary provider (`"litellm"`) and supports secondary alias providers defined statically in `settings.json` under `litellm.providers.<name>`. In containerized environments (Kubernetes, Docker), CI/CD pipelines, and 12-factor deployments, configuring multiple providers solely via environment variables without requiring a pre-populated `settings.json` file is a core requirement.

Environment variable schemes were evaluated against POSIX shell constraints (IEEE Std 1003.1), collision risks with existing top-level variables (`LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_OFFLINE`, `LITELLM_MODELS_DEV`), and Kubernetes SecretKeyRef binding ergonomics.

## Decision
1. **Dedicated Prefix Scanning (`LITELLM_PROVIDER_<NAME>_<KEY>`)**:
   - Secondary providers are discovered dynamically from environment variables matching `LITELLM_PROVIDER_<NAME>_*`.
   - The dedicated `PROVIDER_` namespace token completely isolates secondary provider variables from core top-level `LITELLM_*` configuration settings.
2. **Kebab-Case Default & Canonical Name List (`LITELLM_PROVIDERS`)**:
   - By default, uppercase environment tokens (e.g. `CORP_EAST`) are converted to lowercase kebab-case (`corp-east`).
   - An optional `LITELLM_PROVIDERS="corp-east,staging"` variable preserves lowercase kebab-case and custom underscore provider IDs and deterministic ordering.
3. **Structured JSON Fallback (`LITELLM_PROVIDERS_JSON`)**:
   - A JSON string environment variable provides 1:1 feature parity with `settings.json` for complex nested objects.
4. **Full Feature Parity for Secondary Providers**:
   - Secondary providers support static API keys, `$VAR` templating, `!cmd` command helpers, Google ADC (`USE_GCLOUD_AUTH=1`), OAuth/PKCE (`ENABLE_OAUTH=1`), and interactive CLI login (`/login <alias>`).
5. **Precedence Hierarchy**:
   - `settings.json` takes precedence over environment variables for corresponding fields.
   - Credentials in `auth.json` take precedence over environment variables and settings.

## Consequences
- **Positive**: Seamless deployment in cloud-native / Kubernetes environments; zero collision risk with core `LITELLM_*` variables; supports both flat `.env` files and rich JSON structures.
- **Negative**: Requires scanning `process.env` at initialization; provider token normalization rules must be documented for users.
