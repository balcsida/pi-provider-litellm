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
