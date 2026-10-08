# mockproxy

A lightweight fake LiteLLM proxy for testing Pi/PiG provider extensions end-to-end without network.

## Usage

```bash
go run . -key test-key
```

Flags:
- `-addr` (default `127.0.0.1:0`) – listen address; port 0 picks an ephemeral port
- `-key` (required) – bearer token for authorization
- `-models` (optional) – path to JSON file with custom model definitions
- `-dump` (optional) – detailed request/response logging to stderr: method, path, auth presence, headers, and POST body (pretty-printed JSON)
- `-sso` (default `off`) – login mode: `off`, `cli`, `pkce` or `paste`
- `-sso-pending-polls N` (default `1`) – CLI SSO polls answered `pending` before `ready`
- `-sso-teams` – CLI SSO asks for team selection (`team-a`, `team-b`) before issuing a key
- `-sso-deny` – PKCE authorize redirects with `error=access_denied`

Every authenticated endpoint accepts `-key` and any token the mock issued (opaque `sk-mock-…` strings).

## SSO modes

`off`: `/.well-known/litellm-cli-auth`, `/sso/cli/start` and `/key/generate` answer 404.

`cli`: `POST /sso/cli/start` returns `login_id`, `poll_secret`, `user_code` and a `verification_uri_complete` pointing at `GET /sso/key/generate` (a plain HTML page). `GET /sso/cli/poll/{login_id}` needs the `x-litellm-cli-poll-secret` header (401 otherwise, 400 for an unknown login), answers `pending` for the first N polls, then `ready` with a key, or with `-sso-teams` and no `team_id` query a team selection.

`pkce`: serves `/.well-known/litellm-cli-auth` discovery, `POST /oauth/register`, `GET /oauth/authorize` (302 back to the registered redirect URI with `code` and `state`), `POST /oauth/token` (`authorization_code` with S256 verification, and `refresh_token` with rotation; failures are 400 `invalid_grant`) and `POST /oauth/revoke`.

`paste`: `POST /key/generate` with any bearer token issues a virtual key with a one-hour `expires`; `GET /sso/key/generate` serves the HTML page.

With `-dump`, SSO events are logged to stderr without secrets.

The server prints the listening address to stdout and logs requests to stderr.

## Point a provider at it

Set environment variables:

```bash
export LITELLM_BASE_URL=http://127.0.0.1:8000
export LITELLM_API_KEY=test-key
```

Built-in models: `mock-chat` (Chat Completions), `mock-responses` (Responses API), `mock-claude` (Anthropic Messages).

## Test

```bash
go test ./...
```
