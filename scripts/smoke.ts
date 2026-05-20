// Manual and CI integration smoke test against a real LiteLLM proxy.
// Reads LITELLM_BASE_URL, LITELLM_API_KEY, and LITELLM_SMOKE_MODELS.
// Run: npx tsx scripts/smoke.ts

import { runSmokeFromEnv } from "./smoke-runner.js";

async function main(): Promise<void> {
  const result = await runSmokeFromEnv();
  console.log(`Source: ${result.source}`);
  console.log(`Discovered ${result.discoveredCount} models.`);
  for (const completion of result.completions) {
    console.log(`Smoke OK: ${completion.modelId} -> ${JSON.stringify(completion.content)}`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
