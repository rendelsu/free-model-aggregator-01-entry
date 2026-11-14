# free-model-aggregator-01-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/rendelsu/free-model-aggregator-01-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/rendelsu/free-model-aggregator-01-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions and Anthropic Messages APIs over the best free AI models: each request is routed to the highest-rated free model that meets its requirements, with failover across providers' free tiers when one refuses, and a `--list` of the whole free catalog it found. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

List the catalog it finds (needs the trial brief and env):

```sh
node main.ts --list
```

## How it works

- Reads the trial brief (`MAKEFUN_PROMPT_FILE`): the ten providers, the sealed quality table and the workload manifest.
- Probes every provider's catalog route in its own wire (OpenAI, Anthropic, Gemini) and registers one catalog with the env.
- Routes each request to the best free model that meets its requirements by rating, under a soft per-provider call cap, and fails over to the next-best model on any 402, 429 or 5xx.
- Speaks all six client-api x provider-wire combinations, passing every request through untouched (system prompt, images, tools, stop sequences, max_tokens, JSON mode) and returning the provider's text whole, with its usage and stop reason mapped.
- Deterministic: no wall clock, no unseeded randomness — a replay leaves the same state export.

---

<sub>Entered on [make.fun](https://make.fun/?ref=rkkz4bh8&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
