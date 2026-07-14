# Non-blocking Cache Miss Design

## Goal

Prevent LiteLLM model and MCP discovery from delaying Pi activation when the
model cache is missing or invalid.

## Design

Activation registers each provider immediately from any available cached model
metadata. A cache mismatch still marks the provider for refresh, but does not
discard the cached models or wait for network access. If no cache exists, the
provider starts with no models.

After registration, cache-miss discovery runs in the background through the
existing refresh path. A successful refresh replaces the provider models,
updates costs and cache metadata, and discovers MCP tools. Failures leave the
startup state intact and remain non-fatal.

Commands whose purpose requires current discovery remain synchronous:
`--list-models`, `/login litellm`, and `/litellm-refresh`.

## Testing

Add a startup regression test with a pending discovery request. Extension
activation must complete and register cached models before that request is
resolved. Existing tests continue to cover explicit refresh, login, helper
execution, cache invalidation, and stale background refresh.

## Scope

This change does not optimize eager module imports, change discovery timeouts,
add configuration, or alter the cache format.
