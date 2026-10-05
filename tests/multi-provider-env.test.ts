import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  getProviderDefinitions,
  isValidProviderName,
  parseCanonicalProviderList,
  parseProviderEnvVars,
  parseProvidersJson,
} from "../src/index.js";

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
      baseUrl: "https://corp.example.com",
      apiKeyConfig: "sk-corp",
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
    expect(teamA?.baseUrl).toBe("https://team-a.example.com");
  });

  it("allows overriding provider ID via _NAME variable", () => {
    process.env.LITELLM_PROVIDER_TEAM_A_NAME = "custom_team";
    process.env.LITELLM_PROVIDER_TEAM_A_BASE_URL = "https://team-a.example.com";

    const defs = getProviderDefinitions(undefined);
    expect(defs.find((d) => d.name === "team-a")).toBeUndefined();
    const customTeam = defs.find((d) => d.name === "custom_team");
    expect(customTeam).toBeDefined();
    expect(customTeam?.envPrefix).toBe("LITELLM_PROVIDER_TEAM_A");
    expect(customTeam?.baseUrl).toBe("https://team-a.example.com");
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
      apiKeyConfig: "sk-env",
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
        "BAD-NAME": { baseUrl: "https://bad.example.com" },
      },
    };

    const defs = getProviderDefinitions(settings);
    expect(defs.map((d) => d.name)).toEqual(["litellm"]);
  });
});
