# Null Model Mode Design

## Problem

LiteLLM may return `model_info.mode: null` for local inference backends. Discovery currently treats only an omitted mode as unset, so these otherwise usable models are filtered out.

## Design

Treat `null` and `undefined` as the same unset value in the shared chat-style mode check. Update the response type to reflect LiteLLM's nullable field. Keep explicit non-chat modes, such as `embedding`, filtered.

## Verification

Add one discovery regression test that returns a model with `mode: null`, fails before the production change, and passes afterward. Run the focused discovery tests and the repository check.
