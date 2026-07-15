# Smoke Tests on Pull Requests Design

## Problem

The interactive Pi terminal smoke exits before rendering any discovered model because Terminal Control launches its child with only `PI_CODING_AGENT_DIR`. Parent variables such as `PATH`, `LITELLM_BASE_URL`, and `LITELLM_API_KEY` are not inherited by default.

The LiteLLM smoke workflow also runs only manually, on a schedule, and after relevant pushes to `main`, so pull requests cannot catch smoke regressions before merge.

## Design

Enable environment inheritance for the interactive terminal child while retaining its isolated Pi agent directory. This preserves the workflow-provided LiteLLM configuration without duplicating individual variables.

Add a `pull_request` trigger with the same path filters as the existing `main` push trigger. Keep pull-request runs secret-free by exposing `LITELLM_LICENSE` only for trusted non-PR events; PRs use the existing unlicensed LiteLLM image and skip the Postgres-backed enterprise checks.

## Verification

Add focused static regression assertions before each implementation change. Run the focused workflow test, the repository check, and a clean build. Push the signed feature branch and open a pull request without merging it.
