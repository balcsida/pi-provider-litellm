# Models.dev Cache Design

## Goal

Keep models.dev metadata enrichment available without repeatedly delaying model
discovery, while allowing operators to disable the external metadata source once
their LiteLLM deployment provides trustworthy metadata.

## Configuration

Models.dev enrichment remains enabled by default. Setting
`LITELLM_MODELS_DEV=0` skips both the persistent cache and the models.dev network
request. It does not disable LiteLLM's `/v1/models` fallback; those models still
use the bundled Pi catalog and existing conservative defaults.

The cache lifetime is a fixed 28 days. No additional lifetime setting is added.

## Cache

The extension stores the public models.dev catalog and its fetch timestamp in
`litellm-models-dev.json` under the Pi agent directory. The cache is shared by
LiteLLM provider aliases because the catalog contains no deployment-specific or
credential-specific data. Writes use the repository's existing atomic
temporary-file-and-rename pattern. Missing, malformed, or incompatible cache
files are treated as cache misses.

The existing process-local cache remains, keyed by the persistent cache path so
tests and alternate agent directories do not share state accidentally.

## Discovery Flow

Models.dev remains limited to successful `/v1/models` fallback discovery:

1. With `LITELLM_MODELS_DEV=0`, skip models.dev and map models using the Pi
   catalog and defaults.
2. With a fresh cache, enrich models immediately without a network request.
3. With a stale cache, enrich models immediately from stale data and start one
   deduplicated background refresh. Discovery does not await that refresh.
4. With no usable cache, fetch models.dev synchronously once because no stale
   metadata exists to return.

A successful refresh updates process memory and atomically replaces the disk
cache. A failed background refresh leaves stale data intact. A failed initial
fetch preserves the current Pi-catalog/default fallback. Cache read and write
failures remain non-fatal.

Direct `discoverModels` callers that do not supply an agent cache path retain
the enabled, process-local behavior. Extension discovery paths supply the shared
agent-directory cache path and the environment-controlled enabled flag.

## Testing

Focused discovery tests cover the opt-out, a fresh persistent cache, an initial
cache miss, immediate stale-cache use, successful background replacement,
failed refresh retention, and deduplication of concurrent stale refreshes. An
extension-level test verifies that `LITELLM_MODELS_DEV=0` reaches discovery.

Existing tests continue to cover models.dev metadata priority, Pi catalog
fallback, discovery timeouts, and `/model/info` and `/health` paths.

## Scope

This change does not alter the LiteLLM model cache, expose a configurable cache
lifetime, delete stale data, or couple behavior to a particular LiteLLM issue or
version.
