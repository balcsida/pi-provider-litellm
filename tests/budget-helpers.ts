import { readFileSync } from "node:fs";
import { expect, vi } from "vitest";

export const AUTH = { baseUrl: "https://proxy.example.com", apiKey: "sk-test", headers: { "x-gateway": "g1" } };
export const RESET = Date.parse("2026-11-01T00:00:00Z");

export function fixture(name: string): any {
  return JSON.parse(readFileSync(new URL(`./fixtures/budget/${name}.json`, import.meta.url), "utf8"));
}

export function ok(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

export function status(code: number): Response {
  return new Response(JSON.stringify({ error: "proxy says: secret detail" }), {
    status: code,
    headers: { "content-type": "application/json" },
  });
}

// Maps `pathname + search` to a fresh response (an Error rejects) and returns the recorded requests.
export function mockProxy(
  routes: Record<string, () => Response | Error>,
  expected: { token: string; gateway: string } = { token: "sk-test", gateway: "g1" },
): string[] {
  const seen: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = new URL(String(input));
    const key = url.pathname + url.search;
    seen.push(key);
    const headers = new Headers(init?.headers);
    expect(headers.get("authorization")).toBe(`Bearer ${expected.token}`);
    expect(headers.get("x-gateway")).toBe(expected.gateway);
    const route = routes[key];
    if (!route) return new Response("{}", { status: 404 });
    const result = route();
    if (result instanceof Error) throw result;
    return result;
  });
  return seen;
}
