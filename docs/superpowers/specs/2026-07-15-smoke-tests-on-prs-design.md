# Smoke Tests on Pull Requests Design

## Problem

The interactive Pi terminal smoke exits before rendering any discovered model because Terminal Control launches its child with only `PI_CODING_AGENT_DIR`. Parent variables such as `PATH`, `LITELLM_BASE_URL`, and `LITELLM_API_KEY` are not inherited by default.

Each terminal scenario also uses a fresh agent directory. Because cache-miss discovery is intentionally non-blocking, Pi can launch before that directory contains LiteLLM models and reject the configured provider. Startup failures need terminal logs so CI identifies the failing process boundary.

The LiteLLM smoke workflow also runs only manually, on a schedule, and after relevant pushes to `main`, so pull requests cannot catch smoke regressions before merge.

## Design

Enable environment inheritance for the interactive terminal child while retaining its isolated Pi agent directory. This preserves the workflow-provided LiteLLM configuration without duplicating individual variables.

Before each TUI launch, populate the fresh directory's model cache with the same Pi binary, extension, working directory, and environment. If the initial model never renders, include the terminal logs in the failure without adding noise to successful runs.

Add a `pull_request` trigger with the same path filters as the existing `main` push trigger. Keep pull-request runs secret-free by exposing `LITELLM_LICENSE` only for trusted non-PR events; PRs use the existing unlicensed LiteLLM image and skip the Postgres-backed enterprise checks.

Run auth coverage as two explicit steps against the shared proxy. The community auth step always runs, including on pull requests, with `LITELLM_LICENSE` cleared so it covers missing-token, bad-token, master-key model listing, and master-key chat behavior. A separately named Enterprise auth step runs only when `LITELLM_LICENSE` is configured and covers the licensed virtual-key, admin-route, and SSO behavior. Reusing the existing auth runner keeps the split in workflow orchestration and avoids another proxy job or reusable workflow.

## Verification

Add focused static regression assertions before each implementation change, including the two auth steps and their license handling. Run the focused workflow test, the repository check, and a clean build. Push the signed feature branch and open a pull request without merging it.
