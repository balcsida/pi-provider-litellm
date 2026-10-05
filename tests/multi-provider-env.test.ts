import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AuthInteraction, Provider } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetGcloudTokenCache } from "../src/gcloud-token.js";
import {
  getProviderDefinitions,
  isValidProviderName,
  parseCanonicalProviderList,
  parseProviderEnvVars,
  parseProvidersJson,
  resolveCredentials,
} from "../src/index.js";
import { createPi, loadExtension } from "./test-helpers.js";

async function makeAgentDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "pi-multi-provider-env-"));
}

const TEST_SIGNAL = new AbortController().signal;

function interaction(
  prompt: AuthInteraction["prompt"],
  notify: AuthInteraction["notify"] = vi.fn(),
  signal: AbortSignal = TEST_SIGNAL,
): AuthInteraction & { signal: AbortSignal } {
  return { prompt, notify, signal };
}

function resolveApiKey(provider: Provider, envMap: Record<string, string | undefined> = process.env) {
  return provider.auth.apiKey?.resolve({
    ctx: {
      env: async (name) => envMap[name],
      fileExists: async () => false,
    },
    signal: TEST_SIGNAL,
  });
}

function checkApiKey(provider: Provider, envMap: Record<string, string | undefined> = process.env) {
  return provider.auth.apiKey?.check?.({
    ctx: {
      env: async (name) => envMap[name],
      fileExists: async () => false,
    },
    signal: TEST_SIGNAL,
  });
}

describe("isValidProviderName", () => {
  it("accepts valid lowercase alphanumeric and hyphen/underscore provider names", () => {
    expect(isValidProviderName("corp")).toBe(true);
    expect(isValidProviderName("team-a")).toBe(true);
    expect(isValidProviderName("team_b")).toBe(true);
    expect(isValidProviderName("dev-1_prod")).toBe(true);
    expect(isValidProviderName("a")).toBe(true);
    expect(isValidProviderName("a".repeat(64))).toBe(true);
  });

  it("rejects invalid provider names", () => {
    expect(isValidProviderName("")).toBe(false);
    expect(isValidProviderName("TEAM")).toBe(false);
    expect(isValidProviderName("teamA")).toBe(false);
    expect(isValidProviderName("-team")).toBe(false);
    expect(isValidProviderName("_team")).toBe(false);
    expect(isValidProviderName("team name")).toBe(false);
    expect(isValidProviderName("team@corp")).toBe(false);
    expect(isValidProviderName("a".repeat(65))).toBe(false);
  });

  it("rejects reserved tool and command names", () => {
    expect(isValidProviderName("mcp")).toBe(false);
    expect(isValidProviderName("skills")).toBe(false);
    expect(isValidProviderName("codemode")).toBe(false);
    expect(isValidProviderName("tool_search")).toBe(false);
    expect(isValidProviderName("litellm")).toBe(false);
  });
});

describe("parseCanonicalProviderList", () => {
  it("parses comma- and whitespace-separated provider IDs", () => {
    expect(parseCanonicalProviderList("corp-east, dev")).toEqual(["corp-east", "dev"]);
    expect(parseCanonicalProviderList("corp-east dev team_b")).toEqual(["corp-east", "dev", "team_b"]);
    expect(parseCanonicalProviderList("  team-a,   team_b , team-c  ")).toEqual(["team-a", "team_b", "team-c"]);
  });

  it("handles empty or undefined input", () => {
    expect(parseCanonicalProviderList(undefined)).toEqual([]);
    expect(parseCanonicalProviderList("")).toEqual([]);
    expect(parseCanonicalProviderList("   ")).toEqual([]);
    expect(parseCanonicalProviderList(", , ,")).toEqual([]);
  });
});

describe("parseProvidersJson", () => {
  it("parses valid JSON provider settings map", () => {
    const raw = JSON.stringify({
      "team-a": {
        baseUrl: "https://team-a.example.com",
        apiKey: "sk-team-a",
        displayName: "Team A",
      },
      "team-b": {
        baseUrl: "https://team-b.example.com",
        allowInsecureHttp: true,
      },
    });

    const result = parseProvidersJson(raw);
    expect(result).toBeDefined();
    expect(result?.["team-a"]).toEqual({
      baseUrl: "https://team-a.example.com",
      apiKey: "sk-team-a",
      displayName: "Team A",
    });
    expect(result?.["team-b"]).toEqual({
      baseUrl: "https://team-b.example.com",
      allowInsecureHttp: true,
    });
  });

  it("returns undefined for invalid or non-object JSON", () => {
    expect(parseProvidersJson(undefined)).toBeUndefined();
    expect(parseProvidersJson("")).toBeUndefined();
    expect(parseProvidersJson("{invalid json")).toBeUndefined();
    expect(parseProvidersJson("123")).toBeUndefined();
    expect(parseProvidersJson('"string"')).toBeUndefined();
    expect(parseProvidersJson('["array"]')).toBeUndefined();
  });
});

describe("parseProviderEnvVars", () => {
  it("scans and extracts provider fields by prefix token", () => {
    const env: NodeJS.ProcessEnv = {
      LITELLM_PROVIDER_CORP_EAST_BASE_URL: "https://corp-east.example.com",
      LITELLM_PROVIDER_CORP_EAST_API_KEY: "sk-corp-east",
      LITELLM_PROVIDER_CORP_EAST_DISPLAY_NAME: "Corporate East",
      LITELLM_PROVIDER_CORP_EAST_ALLOW_INSECURE_HTTP: "1",
      LITELLM_PROVIDER_CORP_EAST_USE_GCLOUD_AUTH: "true",
      LITELLM_PROVIDER_CORP_EAST_ENABLE_OAUTH: "1",
      LITELLM_PROVIDER_DEV_BASE_URL: "https://dev.example.com",
      LITELLM_PROVIDER_DEV_API_KEY_HELPER: "dev-key-helper",
      LITELLM_BASE_URL: "https://primary.example.com",
      OTHER_VAR: "ignored",
    };

    const parsed = parseProviderEnvVars(env);
    expect(parsed.CORP_EAST).toEqual({
      token: "CORP_EAST",
      envPrefix: "LITELLM_PROVIDER_CORP_EAST",
      baseUrl: "https://corp-east.example.com",
      apiKey: "sk-corp-east",
      displayName: "Corporate East",
      allowInsecureHttp: true,
      useGcloudTokenAuth: true,
      enableOAuth: true,
    });
    expect(parsed.DEV).toEqual({
      token: "DEV",
      envPrefix: "LITELLM_PROVIDER_DEV",
      baseUrl: "https://dev.example.com",
      apiKeyHelper: "dev-key-helper",
    });
    expect(parsed.LITELLM).toBeUndefined();
    expect(parsed.OTHER).toBeUndefined();
  });

  it("maps boolean fields correctly", () => {
    const env: NodeJS.ProcessEnv = {
      LITELLM_PROVIDER_TEST_ALLOW_INSECURE_HTTP: "true",
      LITELLM_PROVIDER_TEST_USE_GCLOUD_AUTH: "1",
      LITELLM_PROVIDER_TEST_ENABLE_OAUTH: "0",
    };

    const parsed = parseProviderEnvVars(env);
    expect(parsed.TEST.allowInsecureHttp).toBe(true);
    expect(parsed.TEST.useGcloudTokenAuth).toBe(true);
    expect(parsed.TEST.enableOAuth).toBe(false);
  });

  it("parses ENABLE_MCP and MCP_ENABLED boolean variations", () => {
    const env: NodeJS.ProcessEnv = {
      LITELLM_PROVIDER_P1_ENABLE_MCP: "0",
      LITELLM_PROVIDER_P2_ENABLE_MCP: "false",
      LITELLM_PROVIDER_P3_MCP_ENABLED: "0",
      LITELLM_PROVIDER_P4_MCP_ENABLED: "false",
      LITELLM_PROVIDER_P5_ENABLE_MCP: "1",
      LITELLM_PROVIDER_P6_MCP_ENABLED: "true",
    };

    const parsed = parseProviderEnvVars(env);
    expect(parsed.P1.enableMcp).toBe(false);
    expect(parsed.P2.enableMcp).toBe(false);
    expect(parsed.P3.enableMcp).toBe(false);
    expect(parsed.P4.enableMcp).toBe(false);
    expect(parsed.P5.enableMcp).toBe(true);
    expect(parsed.P6.enableMcp).toBe(true);
  });

  it("handles tokens containing reserved substrings like NAME or API", () => {
    const env: NodeJS.ProcessEnv = {
      LITELLM_PROVIDER_MY_NAME_NAME: "custom-name-override",
      LITELLM_PROVIDER_MY_NAME_BASE_URL: "https://myname.example.com",
      LITELLM_PROVIDER_API_CORP_API_KEY_HELPER: "helper-script",
      LITELLM_PROVIDER_API_CORP_BASE_URL: "https://apicorp.example.com",
    };

    const parsed = parseProviderEnvVars(env);
    expect(parsed.MY_NAME).toEqual({
      token: "MY_NAME",
      envPrefix: "LITELLM_PROVIDER_MY_NAME",
      name: "custom-name-override",
      baseUrl: "https://myname.example.com",
    });
    expect(parsed.API_CORP).toEqual({
      token: "API_CORP",
      envPrefix: "LITELLM_PROVIDER_API_CORP",
      apiKeyHelper: "helper-script",
      baseUrl: "https://apicorp.example.com",
    });
  });

  it("parses OIDC env var as JSON or raw string", () => {
    const env: NodeJS.ProcessEnv = {
      LITELLM_PROVIDER_OIDC_JSON_OIDC: '{"clientId": "abc", "issuer": "https://auth.example.com"}',
      LITELLM_PROVIDER_OIDC_STR_OIDC: "https://auth.example.com",
    };

    const parsed = parseProviderEnvVars(env);
    expect(parsed.OIDC_JSON.oidc).toEqual({ clientId: "abc", issuer: "https://auth.example.com" });
    expect(parsed.OIDC_STR.oidc).toBe("https://auth.example.com");
  });
});

describe("getProviderDefinitions with environment variables", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("LITELLM_") || key === "PI_OFFLINE") {
        delete process.env[key];
      }
    }
  });

  afterEach(() => {
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnv)) delete process.env[key];
    }
    Object.assign(process.env, originalEnv);
  });

  it("always includes primary litellm provider with useDefaultEnv: true", () => {
    const defs = getProviderDefinitions(undefined);
    expect(defs.length).toBe(1);
    expect(defs[0]).toMatchObject({
      name: "litellm",
      displayName: "LiteLLM",
      useDefaultEnv: true,
      useGcloudTokenAuth: true,
      enableOAuth: true,
    });
  });

  it("discovers single and multiple providers via prefix-scanned env vars", () => {
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    process.env.LITELLM_PROVIDER_CORP_API_KEY = "sk-corp";
    process.env.LITELLM_PROVIDER_DEV_BASE_URL = "https://dev.example.com";
    process.env.LITELLM_PROVIDER_DEV_API_KEY = "sk-dev";

    const defs = getProviderDefinitions(undefined);
    expect(defs.map((d) => d.name)).toEqual(["litellm", "corp", "dev"]);

    const corp = defs.find((d) => d.name === "corp");
    expect(corp).toMatchObject({
      name: "corp",
      displayName: "corp",
      envPrefix: "LITELLM_PROVIDER_CORP",
      useDefaultEnv: false,
    });
  });

  it("normalizes uppercase tokens with underscores to lowercase hyphens by default", () => {
    process.env.LITELLM_PROVIDER_TEAM_A_BASE_URL = "https://team-a.example.com";
    process.env.LITELLM_PROVIDER_TEAM_A_API_KEY = "sk-team-a";

    const defs = getProviderDefinitions(undefined);
    const teamA = defs.find((d) => d.name === "team-a");
    expect(teamA).toBeDefined();
    expect(teamA?.envPrefix).toBe("LITELLM_PROVIDER_TEAM_A");
  });

  it("allows overriding provider ID via _NAME variable", () => {
    process.env.LITELLM_PROVIDER_TEAM_A_NAME = "custom_team";
    process.env.LITELLM_PROVIDER_TEAM_A_BASE_URL = "https://team-a.example.com";

    const defs = getProviderDefinitions(undefined);
    expect(defs.find((d) => d.name === "team-a")).toBeUndefined();
    const customTeam = defs.find((d) => d.name === "custom_team");
    expect(customTeam).toBeDefined();
    expect(customTeam?.envPrefix).toBe("LITELLM_PROVIDER_TEAM_A");
  });

  it("preserves canonical list IDs specified in LITELLM_PROVIDERS", () => {
    process.env.LITELLM_PROVIDERS = "team-a,team_b";
    process.env.LITELLM_PROVIDER_TEAM_A_BASE_URL = "https://team-a.example.com";
    process.env.LITELLM_PROVIDER_TEAM_B_BASE_URL = "https://team-b.example.com";

    const defs = getProviderDefinitions(undefined);
    expect(defs.map((d) => d.name)).toEqual(["litellm", "team-a", "team_b"]);
    const teamB = defs.find((d) => d.name === "team_b");
    expect(teamB).toBeDefined();
    expect(teamB?.envPrefix).toBe("LITELLM_PROVIDER_TEAM_B");
  });

  it("parses providers from LITELLM_PROVIDERS_JSON", () => {
    process.env.LITELLM_PROVIDERS_JSON = JSON.stringify({
      "json-provider": {
        baseUrl: "https://json.example.com",
        apiKey: "sk-json",
        displayName: "JSON Provider",
      },
    });

    const defs = getProviderDefinitions(undefined);
    const jsonProvider = defs.find((d) => d.name === "json-provider");
    expect(jsonProvider).toMatchObject({
      name: "json-provider",
      displayName: "JSON Provider",
      baseUrl: "https://json.example.com",
      apiKeyConfig: "sk-json",
      useDefaultEnv: false,
    });
  });

  it("merges settings.json with env vars: disk settings override matching fields", () => {
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://env.example.com";
    process.env.LITELLM_PROVIDER_CORP_API_KEY = "sk-env";

    const settings = {
      providers: {
        corp: {
          baseUrl: "https://disk.example.com",
        },
      },
    };

    const defs = getProviderDefinitions(settings);
    const corp = defs.find((d) => d.name === "corp");
    expect(corp).toMatchObject({
      name: "corp",
      baseUrl: "https://disk.example.com",
      envPrefix: "LITELLM_PROVIDER_CORP",
    });
  });

  it("combines non-overlapping disk settings and env vars additively", () => {
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    const settings = {
      providers: {
        diskonly: {
          baseUrl: "https://diskonly.example.com",
        },
      },
    };

    const defs = getProviderDefinitions(settings);
    expect(defs.map((d) => d.name)).toEqual(["litellm", "corp", "diskonly"]);
  });

  it("excludes providers when disk settings specify enabled: false", () => {
    process.env.LITELLM_PROVIDER_DISABLED_BASE_URL = "https://disabled.example.com";
    const settings = {
      providers: {
        disabled: {
          enabled: false,
        },
      },
    };

    const defs = getProviderDefinitions(settings);
    expect(defs.find((d) => d.name === "disabled")).toBeUndefined();
  });

  it("excludes providers when LITELLM_PROVIDERS_JSON specifies enabled: false", () => {
    process.env.LITELLM_PROVIDERS_JSON = JSON.stringify({
      disabled: {
        baseUrl: "https://disabled.example.com",
        enabled: false,
      },
    });

    const defs = getProviderDefinitions(undefined);
    expect(defs.find((d) => d.name === "disabled")).toBeUndefined();
  });

  it("rejects invalid or reserved provider names from env and settings", () => {
    process.env.LITELLM_PROVIDER_MCP_BASE_URL = "https://mcp.example.com";
    process.env.LITELLM_PROVIDER_SKILLS_BASE_URL = "https://skills.example.com";
    process.env.LITELLM_PROVIDER_INVALID_NAME = "Invalid Provider!";
    process.env.LITELLM_PROVIDER_INVALID_BASE_URL = "https://invalid.example.com";

    const settings = {
      providers: {
        tool_search: { baseUrl: "https://tool.example.com" },
        skills: { baseUrl: "https://skills.example.com" },
        mcp: { baseUrl: "https://mcp.example.com" },
        "": { baseUrl: "https://empty.example.com" },
      },
    };

    const defs = getProviderDefinitions(settings);
    expect(defs.map((d) => d.name)).toEqual(["litellm"]);
  });

  it("preserves settings.json aliases like TeamA and team.corp for backwards compatibility", () => {
    const settings = {
      providers: {
        TeamA: {
          baseUrl: "https://teama.example.com",
          apiKey: "sk-teama",
        },
        "team.corp": {
          baseUrl: "https://teamcorp.example.com",
          displayName: "Team Corp",
        },
      },
    };

    const defs = getProviderDefinitions(settings);
    expect(defs.map((d) => d.name)).toEqual(["litellm", "TeamA", "team.corp"]);

    const teamA = defs.find((d) => d.name === "TeamA");
    expect(teamA).toMatchObject({
      name: "TeamA",
      displayName: "TeamA",
      baseUrl: "https://teama.example.com",
      apiKeyConfig: "sk-teama",
    });

    const teamCorp = defs.find((d) => d.name === "team.corp");
    expect(teamCorp).toMatchObject({
      name: "team.corp",
      displayName: "Team Corp",
      baseUrl: "https://teamcorp.example.com",
    });
  });

  it("ensures disabled providers in LITELLM_PROVIDERS_JSON are not re-added by subsequent env steps", () => {
    process.env.LITELLM_PROVIDERS_JSON = JSON.stringify({
      "team-a": {
        baseUrl: "https://json-teama.example.com",
        enabled: false,
      },
      "team-b": {
        baseUrl: "https://json-teamb.example.com",
        enabled: false,
      },
    });

    // Attempt to re-add via canonical list LITELLM_PROVIDERS
    process.env.LITELLM_PROVIDERS = "team-a,team-c";
    process.env.LITELLM_PROVIDER_TEAM_A_BASE_URL = "https://env-teama.example.com";

    // Attempt to re-add via prefix-scanning
    process.env.LITELLM_PROVIDER_TEAM_B_BASE_URL = "https://env-teamb.example.com";
    process.env.LITELLM_PROVIDER_TEAM_C_BASE_URL = "https://env-teamc.example.com";

    const defs = getProviderDefinitions(undefined);
    expect(defs.map((d) => d.name)).toEqual(["litellm", "team-c"]);
    expect(defs.find((d) => d.name === "team-a")).toBeUndefined();
    expect(defs.find((d) => d.name === "team-b")).toBeUndefined();
  });

  it("prevents distinct canonical IDs colliding on token (e.g. team-a vs team_a) from sharing prefix", async () => {
    process.env.LITELLM_PROVIDERS = "team-a,team_a";
    process.env.LITELLM_PROVIDER_TEAM_A_BASE_URL = "https://teama.example.com";
    process.env.LITELLM_PROVIDER_TEAM_A_API_KEY = "sk-team-a";

    const defs = getProviderDefinitions(undefined);
    expect(defs.map((d) => d.name)).toEqual(["litellm", "team-a", "team_a"]);

    const teamHyphen = defs.find((d) => d.name === "team-a");
    expect(teamHyphen).toBeDefined();
    expect(teamHyphen?.envPrefix).toBe("LITELLM_PROVIDER_TEAM_A");

    const teamUnderscore = defs.find((d) => d.name === "team_a");
    expect(teamUnderscore).toBeDefined();
    expect(teamUnderscore?.envPrefix).toBeUndefined();

    const teamHyphenCreds = await resolveCredentials(teamHyphen!);
    expect(teamHyphenCreds.baseUrl).toBe("https://teama.example.com");
    expect(teamHyphenCreds.apiKey).toBe("sk-team-a");

    const teamUnderscoreCreds = await resolveCredentials(teamUnderscore!);
    expect(teamUnderscoreCreds.baseUrl).toBeUndefined();
    expect(teamUnderscoreCreds.apiKey).toBeUndefined();
  });

  it("does not spawn an alias provider from LITELLM_PROVIDER_LITELLM_* variables", () => {
    process.env.LITELLM_PROVIDER_LITELLM_NAME = "team";
    process.env.LITELLM_PROVIDER_LITELLM_ENABLE_MCP = "0";

    const defs = getProviderDefinitions(undefined);
    expect(defs.map((d) => d.name)).toEqual(["litellm"]);

    const litellm = defs.find((d) => d.name === "litellm");
    expect(litellm?.enableMcp).toBe(false);

    expect(defs.find((d) => d.name === "team")).toBeUndefined();
  });

  it("sets enableMcp on ProviderDefinition from environment variables", () => {
    process.env.LITELLM_PROVIDER_DISABLED_MCP_BASE_URL = "https://disabled-mcp.example.com";
    process.env.LITELLM_PROVIDER_DISABLED_MCP_ENABLE_MCP = "false";
    process.env.LITELLM_PROVIDER_ENABLED_MCP_BASE_URL = "https://enabled-mcp.example.com";
    process.env.LITELLM_PROVIDER_ENABLED_MCP_MCP_ENABLED = "1";
    process.env.LITELLM_PROVIDER_DEFAULT_MCP_BASE_URL = "https://default-mcp.example.com";
    process.env.LITELLM_PROVIDER_LITELLM_ENABLE_MCP = "0";

    const defs = getProviderDefinitions(undefined);
    const litellm = defs.find((d) => d.name === "litellm");
    const disabledMcp = defs.find((d) => d.name === "disabled-mcp");
    const enabledMcp = defs.find((d) => d.name === "enabled-mcp");
    const defaultMcp = defs.find((d) => d.name === "default-mcp");

    expect(litellm?.enableMcp).toBe(false);
    expect(disabledMcp?.enableMcp).toBe(false);
    expect(enabledMcp?.enableMcp).toBe(true);
    expect(defaultMcp?.enableMcp).toBeUndefined();
  });

  it("configures or overrides enableMcp via settings.json mcp.enabled or enableMcp", () => {
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    process.env.LITELLM_PROVIDER_CORP_ENABLE_MCP = "1";
    process.env.LITELLM_PROVIDER_TEAM_A_BASE_URL = "https://team-a.example.com";
    process.env.LITELLM_PROVIDER_TEAM_A_ENABLE_MCP = "0";

    const settings = {
      providers: {
        litellm: {
          enableMcp: false,
        },
        corp: {
          mcp: { enabled: false },
        },
        "team-a": {
          enableMcp: true,
        },
        "team-b": {
          baseUrl: "https://team-b.example.com",
          mcp: false,
        },
      },
    };

    const defs = getProviderDefinitions(settings);
    const litellm = defs.find((d) => d.name === "litellm");
    const corp = defs.find((d) => d.name === "corp");
    const teamA = defs.find((d) => d.name === "team-a");
    const teamB = defs.find((d) => d.name === "team-b");

    expect(litellm?.enableMcp).toBe(false);
    expect(corp?.enableMcp).toBe(false);
    expect(teamA?.enableMcp).toBe(true);
    expect(teamB?.enableMcp).toBe(false);
  });
});

describe("Per-provider credential and endpoint resolution", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    resetGcloudTokenCache();
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("LITELLM_") || key.startsWith("GOOGLE_") || key === "PI_OFFLINE") {
        delete process.env[key];
      }
    }
  });

  afterEach(() => {
    vi.restoreAllMocks();
    resetGcloudTokenCache();
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnv)) delete process.env[key];
    }
    Object.assign(process.env, originalEnv);
  });

  it("uses scoped _BASE_URL and _API_KEY for secondary providers", async () => {
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    process.env.LITELLM_PROVIDER_CORP_API_KEY = "sk-corp";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const corp = pi.providers.find((p) => p.id === "corp");
    expect(corp).toBeDefined();

    const checkResult = await checkApiKey(corp!);
    expect(checkResult).toEqual({
      type: "api_key",
      source: "$LITELLM_PROVIDER_CORP_API_KEY",
    });

    const authResult = await resolveApiKey(corp!);
    expect(authResult).toEqual({
      auth: {
        apiKey: "sk-corp",
        headers: undefined,
        baseUrl: "https://corp.example.com",
      },
      env: {
        LITELLM_BASE_URL: "https://corp.example.com",
      },
      source: "$LITELLM_PROVIDER_CORP_API_KEY",
    });
  });

  it("does NOT leak or fallback to global LITELLM_BASE_URL and LITELLM_API_KEY for secondary providers", async () => {
    process.env.LITELLM_BASE_URL = "https://primary.example.com";
    process.env.LITELLM_API_KEY = "sk-primary";
    process.env.LITELLM_PROVIDERS = "corp";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const corp = pi.providers.find((p) => p.id === "corp");
    expect(corp).toBeDefined();

    // Corp has no scoped BASE_URL or API_KEY, so check and resolve must return undefined
    const checkResult = await checkApiKey(corp!);
    expect(checkResult).toBeUndefined();

    const authResult = await resolveApiKey(corp!);
    expect(authResult).toBeUndefined();

    // Primary provider still resolves its global credentials
    const litellm = pi.providers.find((p) => p.id === "litellm");
    expect(litellm).toBeDefined();
    const litellmAuth = await resolveApiKey(litellm!);
    expect(litellmAuth).toMatchObject({
      auth: {
        apiKey: "sk-primary",
        baseUrl: "https://primary.example.com",
      },
      source: "LITELLM_API_KEY",
    });
  });

  it("lazily executes scoped _API_KEY_HELPER command for secondary providers", async () => {
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    process.env.LITELLM_PROVIDER_CORP_API_KEY_HELPER = "echo sk-corp-helper";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const corp = pi.providers.find((p) => p.id === "corp");
    expect(corp).toBeDefined();

    const checkResult = await checkApiKey(corp!);
    expect(checkResult).toEqual({
      type: "api_key",
      source: "$LITELLM_PROVIDER_CORP_API_KEY_HELPER",
    });

    const authResult = await resolveApiKey(corp!);
    expect(authResult).toEqual({
      auth: {
        apiKey: "sk-corp-helper",
        headers: undefined,
        baseUrl: "https://corp.example.com",
      },
      env: {
        LITELLM_BASE_URL: "https://corp.example.com",
      },
      source: "$LITELLM_PROVIDER_CORP_API_KEY_HELPER",
    });
  });

  it("resolves and passes scoped _HEADERS JSON for secondary providers", async () => {
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    process.env.LITELLM_PROVIDER_CORP_API_KEY = "sk-corp";
    process.env.LITELLM_PROVIDER_CORP_HEADERS = JSON.stringify({
      "X-Custom-Tenant": "tenant-corp",
      "X-Secret-Env": "$DYNAMIC_SECRET",
    });
    process.env.DYNAMIC_SECRET = "secret-123";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const corp = pi.providers.find((p) => p.id === "corp");
    expect(corp).toBeDefined();

    const authResult = await resolveApiKey(corp!);
    expect(authResult?.auth.headers).toEqual({
      "X-Custom-Tenant": "tenant-corp",
      "X-Secret-Env": "secret-123",
    });
  });

  it("enables Google ADC when _USE_GCLOUD_AUTH is set to 1 for secondary providers", async () => {
    const agentDir = await makeAgentDir();
    const adcPath = join(agentDir, "adc.json");
    await writeFile(
      adcPath,
      JSON.stringify({
        type: "authorized_user",
        client_id: "client-id",
        client_secret: "client-secret",
        refresh_token: "refresh-token",
      }),
      "utf8",
    );
    process.env.GOOGLE_APPLICATION_CREDENTIALS = adcPath;
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    process.env.LITELLM_PROVIDER_CORP_USE_GCLOUD_AUTH = "1";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input instanceof Request ? input.url : input);
      if (url === "https://oauth2.googleapis.com/token") {
        return new Response(JSON.stringify({ access_token: "ya29.corp-minted", expires_in: 3600 }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      throw new Error(`unexpected fetch ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const corp = pi.providers.find((p) => p.id === "corp");
    expect(corp).toBeDefined();

    const checkResult = await checkApiKey(corp!);
    expect(checkResult).toEqual({ type: "api_key", source: "gcloud ADC" });

    const authResult = await resolveApiKey(corp!);
    expect(authResult).toEqual({
      auth: {
        apiKey: "ya29.corp-minted",
        headers: undefined,
        baseUrl: "https://corp.example.com",
      },
      env: {
        LITELLM_BASE_URL: "https://corp.example.com",
      },
      source: "gcloud ADC",
    });
  });

  it("does NOT use Google ADC for secondary providers when _USE_GCLOUD_AUTH is not set", async () => {
    const agentDir = await makeAgentDir();
    const adcPath = join(agentDir, "adc.json");
    await writeFile(
      adcPath,
      JSON.stringify({
        type: "authorized_user",
        client_id: "client-id",
        client_secret: "client-secret",
        refresh_token: "refresh-token",
      }),
      "utf8",
    );
    process.env.GOOGLE_APPLICATION_CREDENTIALS = adcPath;
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const corp = pi.providers.find((p) => p.id === "corp");
    expect(corp).toBeDefined();

    const checkResult = await checkApiKey(corp!);
    expect(checkResult).toBeUndefined();

    const authResult = await resolveApiKey(corp!);
    expect(authResult).toBeUndefined();
  });

  it("enables OAuth descriptor on provider auth when _ENABLE_OAUTH is set to 1", async () => {
    process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
    process.env.LITELLM_PROVIDER_CORP_ENABLE_OAUTH = "1";
    process.env.LITELLM_PROVIDER_DEV_BASE_URL = "https://dev.example.com";
    process.env.LITELLM_PROVIDER_DEV_ENABLE_OAUTH = "0";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const corp = pi.providers.find((p) => p.id === "corp");
    expect(corp).toBeDefined();
    expect(corp?.auth.oauth).toBeDefined();
    expect(corp?.auth.oauth?.name).toBe("LiteLLM SSO");

    const dev = pi.providers.find((p) => p.id === "dev");
    expect(dev).toBeDefined();
    expect(dev?.auth.oauth).toBeUndefined();
  });

  describe("resolveCredentials directly", () => {
    it("resolves scoped baseUrl and apiKey for secondary provider definition", async () => {
      process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
      process.env.LITELLM_PROVIDER_CORP_API_KEY = "sk-corp";

      const creds = await resolveCredentials({
        name: "corp",
        displayName: "corp",
        envPrefix: "LITELLM_PROVIDER_CORP",
        useDefaultEnv: false,
        useGcloudTokenAuth: false,
        enableOAuth: false,
        allowInsecureHttp: false,
      });

      expect(creds).toEqual({
        baseUrl: "https://corp.example.com",
        apiKey: "sk-corp",
        apiKeyConfig: "$LITELLM_PROVIDER_CORP_API_KEY",
        apiKeyFromGcloudAdc: false,
      });
    });

    it("does not fall back to global LITELLM_BASE_URL and LITELLM_API_KEY", async () => {
      process.env.LITELLM_BASE_URL = "https://primary.example.com";
      process.env.LITELLM_API_KEY = "sk-primary";

      const creds = await resolveCredentials({
        name: "corp",
        displayName: "corp",
        envPrefix: "LITELLM_PROVIDER_CORP",
        useDefaultEnv: false,
        useGcloudTokenAuth: false,
        enableOAuth: false,
        allowInsecureHttp: false,
      });

      expect(creds).toEqual({
        baseUrl: undefined,
        apiKey: undefined,
        apiKeyConfig: undefined,
        apiKeyFromGcloudAdc: false,
      });
    });

    it("resolves scoped API key helper command", async () => {
      process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
      process.env.LITELLM_PROVIDER_CORP_API_KEY_HELPER = "echo sk-helper-out";

      const creds = await resolveCredentials({
        name: "corp",
        displayName: "corp",
        envPrefix: "LITELLM_PROVIDER_CORP",
        useDefaultEnv: false,
        useGcloudTokenAuth: false,
        enableOAuth: false,
        allowInsecureHttp: false,
      });

      expect(creds).toEqual({
        baseUrl: "https://corp.example.com",
        apiKey: "sk-helper-out",
        apiKeyConfig: "!echo sk-helper-out",
        apiKeyFromGcloudAdc: false,
      });
    });
  });

  describe("interactive login for secondary providers", () => {
    it("provides apiKey.login on secondary providers", async () => {
      process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);

      const corp = pi.providers.find((p) => p.id === "corp");
      expect(corp).toBeDefined();
      expect(corp?.auth.apiKey?.login).toBeDefined();
    });

    it("offers the provider-scoped base URL in knownBaseUrl for /login <alias>", async () => {
      process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);

      const corp = pi.providers.find((p) => p.id === "corp");
      expect(corp).toBeDefined();

      let offeredOptions: readonly { id: string; label: string }[] | undefined;
      const prompt = vi.fn(async (options: Parameters<AuthInteraction["prompt"]>[0]) => {
        if ("options" in options && options.options) {
          offeredOptions = options.options;
          return options.options[0]!.id;
        }
        return "sk-corp-key";
      });

      const credential = await corp?.auth.apiKey?.login?.(interaction(prompt));

      expect(offeredOptions).toEqual([
        { id: "https://corp.example.com", label: "https://corp.example.com ($LITELLM_PROVIDER_CORP_BASE_URL)" },
        { id: "enter-a-different-url", label: "Enter a different URL…" },
      ]);
      expect(credential).toEqual({
        type: "api_key",
        key: "sk-corp-key",
        env: { LITELLM_BASE_URL: "https://corp.example.com" },
      });
    });

    it("completes login for a secondary provider with custom URL and API key", async () => {
      process.env.LITELLM_PROVIDER_CORP_BASE_URL = "https://corp.example.com";
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);

      const corp = pi.providers.find((p) => p.id === "corp");
      const prompt = vi.fn(async (options: Parameters<AuthInteraction["prompt"]>[0]) => {
        if ("options" in options && options.options) {
          return "enter-a-different-url";
        }
        if ("placeholder" in options) {
          return "https://custom-corp.example.com";
        }
        return "sk-custom-corp";
      });

      const credential = await corp?.auth.apiKey?.login?.(interaction(prompt));
      expect(credential).toEqual({
        type: "api_key",
        key: "sk-custom-corp",
        env: { LITELLM_BASE_URL: "https://custom-corp.example.com" },
      });
    });

    it("drops existing MCP registration for <alias> on login start", async () => {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = String(input);
        if (url.endsWith("/model/info")) return Response.json({ data: [] });
        throw new Error(`unexpected URL: ${url}`);
      });
      const agentDir = await makeAgentDir();
      await writeFile(
        join(agentDir, "auth.json"),
        JSON.stringify({
          corp: {
            type: "api_key",
            key: "sk-stored",
            env: { LITELLM_BASE_URL: "https://corp.example.com" },
          },
        }),
        "utf8",
      );
      await writeFile(
        join(agentDir, "settings.json"),
        JSON.stringify({
          litellm: {
            providers: {
              corp: {
                baseUrl: "https://corp.example.com",
              },
            },
          },
        }),
        "utf8",
      );

      const extension = await loadExtension(agentDir);
      const pi = createPi();
      await extension(pi);

      expect(pi.mcpServers.has("corp")).toBe(true);

      const registeredWhenPrompted: boolean[] = [];
      const prompt = vi.fn(async (options: Parameters<AuthInteraction["prompt"]>[0]) => {
        registeredWhenPrompted.push(pi.mcpServers.has("corp"));
        if ("options" in options && options.options) {
          return options.options[0]!.id;
        }
        return "sk-new-key";
      });

      const corp = pi.providers.find((p) => p.id === "corp");
      await corp?.auth.apiKey?.login?.(interaction(prompt));

      expect(registeredWhenPrompted).toEqual([false, false]);
      expect(pi.mcpServers.has("corp")).toBe(false);

      for (const handler of pi.handlers.get("before_agent_start") ?? []) {
        await handler({ systemPrompt: "" }, {});
      }
      expect(pi.mcpServers.has("corp")).toBe(true);
    });

    it("does not call syncMcpServer in complete() and does not re-enable disabled MCP", async () => {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = String(input);
        if (url.endsWith("/model/info")) return Response.json({ data: [] });
        throw new Error(`unexpected URL: ${url}`);
      });
      const agentDir = await makeAgentDir();
      await writeFile(
        join(agentDir, "auth.json"),
        JSON.stringify({
          corp: {
            type: "api_key",
            key: "sk-stored",
            env: { LITELLM_BASE_URL: "https://corp.example.com" },
          },
        }),
        "utf8",
      );
      await writeFile(
        join(agentDir, "settings.json"),
        JSON.stringify({
          litellm: {
            providers: {
              corp: {
                baseUrl: "https://corp.example.com",
                enableMcp: false,
              },
            },
          },
        }),
        "utf8",
      );

      const extension = await loadExtension(agentDir);
      const pi = createPi();
      await extension(pi);

      expect(pi.mcpServers.has("corp")).toBe(false);

      const prompt = vi.fn(async (options: Parameters<AuthInteraction["prompt"]>[0]) => {
        if ("options" in options && options.options) {
          return options.options[0]!.id;
        }
        return "sk-new-key";
      });

      const corp = pi.providers.find((p) => p.id === "corp");
      await corp?.auth.apiKey?.login?.(interaction(prompt));

      expect(pi.mcpServers.has("corp")).toBe(false);

      for (const handler of pi.handlers.get("before_agent_start") ?? []) {
        await handler({ systemPrompt: "" }, {});
      }
      expect(pi.mcpServers.has("corp")).toBe(false);
    });
  });
});

describe("Per-provider MCP registration toggle", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("LITELLM_") || key === "PI_OFFLINE") {
        delete process.env[key];
      }
    }
  });

  afterEach(() => {
    vi.restoreAllMocks();
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnv)) delete process.env[key];
    }
    Object.assign(process.env, originalEnv);
  });

  it("does not register MCP server for a provider when enableMcp is false", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return Response.json({ data: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: {
          type: "api_key",
          key: "sk-litellm",
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        },
        "team-enabled": {
          type: "api_key",
          key: "sk-enabled",
          env: { LITELLM_BASE_URL: "https://enabled.example.com" },
        },
        "team-disabled-env": {
          type: "api_key",
          key: "sk-disabled-env",
          env: { LITELLM_BASE_URL: "https://disabled-env.example.com" },
        },
        "team-disabled-settings": {
          type: "api_key",
          key: "sk-disabled-settings",
          env: { LITELLM_BASE_URL: "https://disabled-settings.example.com" },
        },
      }),
      "utf8",
    );

    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "team-enabled": {
              baseUrl: "https://enabled.example.com",
            },
            "team-disabled-env": {
              baseUrl: "https://disabled-env.example.com",
            },
            "team-disabled-settings": {
              baseUrl: "https://disabled-settings.example.com",
              mcp: { enabled: false },
            },
          },
        },
      }),
      "utf8",
    );

    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-litellm";
    process.env.LITELLM_PROVIDER_TEAM_DISABLED_ENV_ENABLE_MCP = "false";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.mcpServers.has("litellm")).toBe(true);
    expect(pi.mcpServers.has("team-enabled")).toBe(true);
    expect(pi.mcpServers.has("team-disabled-env")).toBe(false);
    expect(pi.mcpServers.has("team-disabled-settings")).toBe(false);
  });

  it("does not register MCP server for primary litellm provider when disabled via env or settings", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return Response.json({ data: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: {
          type: "api_key",
          key: "sk-litellm",
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        },
      }),
      "utf8",
    );

    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            litellm: {
              enableMcp: false,
            },
          },
        },
      }),
      "utf8",
    );

    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-litellm";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.mcpServers.has("litellm")).toBe(false);
  });

  it("does not register MCP server for primary litellm provider when disabled via env var", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return Response.json({ data: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: {
          type: "api_key",
          key: "sk-litellm",
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        },
      }),
      "utf8",
    );

    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-litellm";
    process.env.LITELLM_PROVIDER_LITELLM_ENABLE_MCP = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.mcpServers.has("litellm")).toBe(false);
  });
});
