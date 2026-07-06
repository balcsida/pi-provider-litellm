import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CACHE_TTL_MS, getGcloudToken, resetGcloudTokenCache } from "../src/gcloud-token.js";

const ORIGINAL_ENV = {
  APPDATA: process.env.APPDATA,
  GOOGLE_APPLICATION_CREDENTIALS: process.env.GOOGLE_APPLICATION_CREDENTIALS,
  HOME: process.env.HOME,
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function writeAdcFile(body: unknown): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-gcloud-"));
  const path = join(dir, "adc.json");
  await writeFile(path, JSON.stringify(body), "utf8");
  return path;
}

afterEach(() => {
  for (const [key, value] of Object.entries(ORIGINAL_ENV)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  resetGcloudTokenCache();
  vi.restoreAllMocks();
});

describe("getGcloudToken", () => {
  it("exchanges authorized_user ADC credentials for an access token", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { access_token: "ya29.token" }));

    await expect(getGcloudToken()).resolves.toBe("ya29.token");

    expect(fetchMock).toHaveBeenCalledWith(
      "https://oauth2.googleapis.com/token",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: expect.stringContaining("grant_type=refresh_token"),
      }),
    );
  });

  it("uses the cached token until the TTL expires", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "first-token" }))
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "second-token" }));

    const now = new Date("2026-06-10T12:00:00.000Z").getTime();
    vi.spyOn(Date, "now").mockReturnValue(now);
    await expect(getGcloudToken()).resolves.toBe("first-token");
    await expect(getGcloudToken()).resolves.toBe("first-token");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    vi.mocked(Date.now).mockReturnValue(now + CACHE_TTL_MS + 1);
    await expect(getGcloudToken()).resolves.toBe("second-token");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not reuse a cached token after the ADC identity changes", async () => {
    const firstPath = await writeAdcFile({
      type: "authorized_user",
      client_id: "first-client",
      client_secret: "first-secret",
      refresh_token: "first-refresh",
    });
    const secondPath = await writeAdcFile({
      type: "authorized_user",
      client_id: "second-client",
      client_secret: "second-secret",
      refresh_token: "second-refresh",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "first-token" }))
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "second-token" }));
    vi.spyOn(Date, "now").mockReturnValue(new Date("2026-06-10T12:00:00.000Z").getTime());

    process.env.GOOGLE_APPLICATION_CREDENTIALS = firstPath;
    await expect(getGcloudToken()).resolves.toBe("first-token");

    process.env.GOOGLE_APPLICATION_CREDENTIALS = secondPath;
    await expect(getGcloudToken()).resolves.toBe("second-token");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("returns null and warns when no ADC file exists", async () => {
    delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
    delete process.env.APPDATA;
    process.env.HOME = await mkdtemp(join(tmpdir(), "pi-litellm-no-adc-"));
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("No Google ADC file found"));
  });

  it("returns null and warns for service account credentials", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({ type: "service_account" });
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("Service account credentials are not supported"));
  });

  it("returns null when the token exchange fails", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("invalid_grant", { status: 400 }));
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("Token exchange failed"));
  });
});
