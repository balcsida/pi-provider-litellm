# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-28

### Added

- `pi-package` keyword so the package surfaces in the [pi.dev/packages](https://pi.dev/packages) gallery.

### Changed

- README install instructions now use `pi install npm:@balcsida/pi-provider-litellm` (and `pi -e` for ephemeral trials), with the source-clone path kept as a fallback.

## [0.1.0] - 2026-04-27

### Added

- Initial release.
- Discovers models from a self-hosted LiteLLM proxy via `/model/info` with fallback to `/v1/models` on 401/403/404.
- Sha256-fingerprinted disk cache with atomic writes.
- `/login litellm` interactive login (prompts for base URL + API key).
- `/litellm-refresh` slash command to force re-fetch.
- `LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_OFFLINE`, `LITELLM_DISCOVERY_TIMEOUT_MS` environment variables.
- `compat.cacheControlFormat: "anthropic"` for both `anthropic/*`-prefixed model IDs and bare Claude aliases (`claude-*`, `opus-*`, `sonnet-*`, `haiku-*`, case-insensitive). Common LiteLLM deployments alias Claude models without the `anthropic/` prefix; without this flag, prompt caching silently no-ops on Claude through the proxy.
