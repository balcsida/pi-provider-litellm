import { afterEach, describe, expect, it, vi } from "vitest";
import { parseProxyVersion, probeProxyVersion, proxyVersionAtLeast } from "../src/proxy-version.js";

afterEach(() => {
  vi.restoreAllMocks();
});

function probeResponse(status: number, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify({ error: { message: "no healthy deployments" } }), { status, headers });
}

describe("parseProxyVersion", () => {
  it.each([
    ["1.102.0", { major: 1, minor: 102, patch: 0, prerelease: false }],
    ["v1.103.0", { major: 1, minor: 103, patch: 0, prerelease: false }],
    [" 1.103.1 ", { major: 1, minor: 103, patch: 1, prerelease: false }],
    ["1.103.0-rc.1", { major: 1, minor: 103, patch: 0, prerelease: true }],
    ["1.103.0rc1", { major: 1, minor: 103, patch: 0, prerelease: true }],
    ["1.103.0.dev2", { major: 1, minor: 103, patch: 0, prerelease: true }],
    ["1.103.0.post1", { major: 1, minor: 103, patch: 0, prerelease: true }],
  ])("reads %j", (value, expected) => {
    expect(parseProxyVersion(value)).toEqual(expected);
  });

  it.each([
    [""],
    ["latest"],
    ["1.103"],
    ["1"],
    ["one.two.three"],
    [`1.103.0-${"x".repeat(80)}`],
    ["999999.0.0"],
    [undefined],
    [null],
    [1.103],
    [["1.103.0"]],
  ])("withholds a version from %j", (value) => {
    expect(parseProxyVersion(value)).toBeUndefined();
  });
});

describe("proxyVersionAtLeast", () => {
  it.each([
    ["1.103.0", true],
    ["1.103.1", true],
    ["1.104.0", true],
    ["2.0.0", true],
    ["1.104.0-rc.1", true],
    ["1.102.1", false],
    ["1.102.99", false],
    ["0.999.999", false],
    ["1.103.0-rc.1", false],
    ["1.103.0.dev2", false],
  ])("orders %s against 1.103.0", (value, expected) => {
    expect(proxyVersionAtLeast(parseProxyVersion(value), [1, 103, 0])).toBe(expected);
  });

  it("treats an unknown version as older than any minimum", () => {
    expect(proxyVersionAtLeast(undefined, [0, 0, 0])).toBe(false);
  });
});

describe("probeProxyVersion", () => {
  it("reads the version from the error reply to a made-up Responses id", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(probeResponse(400, { "x-litellm-version": "1.103.0" }));

    await expect(probeProxyVersion("https://proxy.test", "sk-test", { headers: { "x-tenant": "a" } })).resolves.toEqual(
      { major: 1, minor: 103, patch: 0, prerelease: false },
    );

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(String(url)).toBe("https://proxy.test/v1/responses/resp_version_probe");
    expect(init?.method ?? "GET").toBe("GET");
    expect(init?.headers).toMatchObject({ "x-tenant": "a", Authorization: "Bearer sk-test" });
  });

  it.each([
    ["a reply without the header", () => Promise.resolve(probeResponse(400))],
    ["an unreadable header", () => Promise.resolve(probeResponse(400, { "x-litellm-version": "latest" }))],
    ["a refused connection", () => Promise.reject(new TypeError("fetch failed"))],
    ["an aborted request", () => Promise.reject(new DOMException("aborted", "AbortError"))],
  ])("withholds a version on %s", async (_label, respond) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(respond);

    await expect(probeProxyVersion("https://proxy.test", "sk-test", {})).resolves.toBeUndefined();
  });

  it("does not read the reply body", async () => {
    const response = probeResponse(400, { "x-litellm-version": "1.103.0" });
    const json = vi.spyOn(response, "json");
    const text = vi.spyOn(response, "text");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response);

    await probeProxyVersion("https://proxy.test", "sk-test", {});

    expect(json).not.toHaveBeenCalled();
    expect(text).not.toHaveBeenCalled();
  });
});
