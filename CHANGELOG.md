## 0.5.5

- bump `agent-eval` to 0.1.55 and `litellm` to 1.97.0 so submission scoring can price calls routed through an `osd-proxy/` model prefix. agent-eval 0.1.55 registers the litellm v1.97.0 price map, which includes `gpt-5.6-sol`, and translates `osd-proxy/gpt-5.6-sol` to it. As in 0.5.4, both pins move together: agent-eval for the prices, litellm for provider resolution. litellm 1.97.0 is the newest release that does not unconditionally require `boto3>=1.43.1`, which conflicts with the scorer's `aiobotocore` 2.x stack.

## 0.5.4

- bump `agent-eval[leaderboard]` to 0.1.51 and `litellm` to 1.88.1 so submission scoring can price newer models such as `gemini-3.5-flash`. agent-eval 0.1.51 (agent-eval#84) registers the litellm v1.88.1 price map, but `agenteval.log.compute_model_cost` prices each call via `litellm.cost_per_token(usage_object=...)`, which resolves the model's provider from the *installed* litellm's model registry — not from the registered price map. litellm 1.82.3 does not know `gemini-3.5-flash`, so `cost_per_token` raises `LLM Provider NOT provided`, which makes `compute_model_cost` null the entire task cost. Both pins must therefore move together: agent-eval for the prices, litellm for provider resolution.

## 0.4.2

- update retired Anthropic and Gemini scorer/default model IDs to current Claude Sonnet 4.6 and Gemini 3 variants

## 0.3.1

- pick up fix for SSE bug which sometimes caused MCP calls to freeze

## 0.3.0

- update paper-finder task instructions to better reflect scoring

## 0.2.0

- original public release
