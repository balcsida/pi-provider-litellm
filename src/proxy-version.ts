import type { DiscoveryOptions } from "./types.js";

export interface ProxyVersion {
  major: number;
  minor: number;
  patch: number;
  // Any suffix (rc, dev, post, local build). It orders before the release it names,
  // so a pre-release never claims a fix that only the release is known to carry.
  prerelease: boolean;
}

export type ProxyVersionFloor = readonly [major: number, minor: number, patch: number];

const DEFAULT_TIMEOUT_MS = 5000;
const MAX_VERSION_LENGTH = 64;
const VERSION_PATTERN = /^v?(\d{1,5})\.(\d{1,5})\.(\d{1,5})(.*)$/s;
const VERSION_HEADER = "x-litellm-version";
// LiteLLM stamps its version on inference routes only; `/model/info`, `/v1/models`
// and `/health/*` do not carry it. A made-up Responses id names no deployment, so
// the read is answered with an error that still carries the header and calls no model.
const VERSION_PROBE_PATH = "/v1/responses/resp_version_probe";

// The value is proxy-supplied: bound it before matching and keep only the numbers.
export function parseProxyVersion(value: unknown): ProxyVersion | undefined {
  if (typeof value !== "string" || value.length > MAX_VERSION_LENGTH) return undefined;
  const match = VERSION_PATTERN.exec(value.trim());
  if (!match) return undefined;
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    prerelease: match[4] !== "",
  };
}

// An unknown version satisfies no floor, so a version-gated workaround stays on
// whenever the proxy does not say what it is.
export function proxyVersionAtLeast(version: ProxyVersion | undefined, floor: ProxyVersionFloor): boolean {
  if (!version) return false;
  const actual = [version.major, version.minor, version.patch];
  for (const [index, required] of floor.entries()) {
    if (actual[index] !== required) return actual[index]! > required;
  }
  return !version.prerelease;
}

// Best effort by design: the header rides on an error path LiteLLM does not
// document, so a missing header, an unreachable proxy or a timeout all mean
// "unknown" and never fail discovery.
export async function probeProxyVersion(
  base: string,
  apiKey: string,
  options: Pick<DiscoveryOptions, "timeoutMs" | "signal" | "headers">,
): Promise<ProxyVersion | undefined> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  try {
    const response = await fetch(`${base}${VERSION_PROBE_PATH}`, {
      headers: { ...options.headers, Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
      signal: options.signal
        ? AbortSignal.any([options.signal, AbortSignal.timeout(timeoutMs)])
        : AbortSignal.timeout(timeoutMs),
    });
    await response.body?.cancel().catch(() => undefined);
    return parseProxyVersion(response.headers.get(VERSION_HEADER));
  } catch {
    return undefined;
  }
}
