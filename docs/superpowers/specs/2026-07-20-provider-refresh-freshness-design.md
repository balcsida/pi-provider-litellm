# Provider Refresh Freshness Design

## Problem

Pi 0.80.10 initializes its footer by refreshing registered providers after the
extension's `session_start` handler has already scheduled model discovery. The
LiteLLM provider callback currently ignores Pi's freshness and `force`
semantics, so the same model and MCP discovery runs twice and prints two full
progress sequences.

## Design

Keep the existing non-blocking `session_start` refresh. Before handling Pi's
provider refresh callback, return the current models when the extension has a
fresh cache and Pi did not request a forced refresh. Continue refreshing when
the cache is missing, invalid, stale, or when `force` is true.

The explicit `/litellm-refresh` command remains unchanged because it calls the
shared refresh operation directly. No new setting, dependency, or logging
layer is needed.

## Verification

Add a regression test that invokes Pi's provider callback after a completed
session refresh and verifies that it performs no second request. Also verify
that a forced callback still performs discovery. Run the focused test, the
repository check, a clean build, and an interactive Pi startup smoke test.

## CI Follow-up: Terminal Editor Readiness

Pi's asynchronous provider autocomplete can consume Enter to accept `litellm`
without submitting `/login litellm`. Keep the 90-second failure bound, wait for
the exact command and LiteLLM provider suggestion, then dismiss autocomplete
with Escape before pressing Enter. Do not apply visible-text checks to prompt
values because credentials may be masked.
