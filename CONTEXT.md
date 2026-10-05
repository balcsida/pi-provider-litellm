# Glossary & Domain Terms

## Core Concepts

### Primary Provider (`"litellm"`)
The default, singleton LiteLLM provider instance registered in Pi. Consumes global `LITELLM_*` environment variables (`LITELLM_BASE_URL`, `LITELLM_API_KEY`, etc.), hosts the LiteLLM Skills Gateway, and serves as the baseline proxy endpoint.

### Secondary / Alias Provider
Additional LiteLLM provider instances registered in Pi (e.g. `corp-east`, `staging-local`, `backup-proxy`). Configured via `settings.json` (`litellm.providers.<name>`) or provider-scoped environment variables (`LITELLM_PROVIDER_<NAME>_*`).

### Provider Definition (`ProviderDefinition`)
The internal normalized runtime model describing a LiteLLM provider's identity, endpoint (`baseUrl`), authentication rules (`apiKeyConfig`, `useGcloudTokenAuth`, `enableOAuth`), custom headers, and environment variable bindings.

### Prefix Scanning (`LITELLM_PROVIDER_<NAME>_*`)
Dynamic discovery mechanism that scans `process.env` for uppercase provider tokens (e.g., `LITELLM_PROVIDER_TEAM_A_BASE_URL`) to instantiate secondary providers without manual configuration files.

### Host Pinning
Security enforcement ensuring that all model execution and discovery requests strictly target the host validated by the credential root (`requireCredentialRoot`), preventing malicious or off-catalog model definitions from exfiltrating credentials to unverified hosts.
