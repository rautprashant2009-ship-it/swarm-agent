---
name: swarm-agent
description: "Seamlessly command and orchestrate the local Swarm Agent gateway and autonomous multi-agent runtime. Query 7 frontier models (DeepSeek-V4-Pro, Kimi-K3, GLM-5.3, Qwen-3.8, Gemini-3.8-Flash-High, Muse-Spark, Groq, Cerebras) with transparent 429 quota failover, apply context compression, execute guarded host commands, and trigger 3-tier nested self-healing convergence loops."
---

# Swarm Agent Integration for Antigravity

This skill enables Antigravity agents to orchestrate, delegate to, and leverage the local **Swarm Agent** multi-provider runtime located at `C:\Users\Admin\.gemini\antigravity\scratch\swarm-agent`.

## Core Capabilities

1. **Frontier Model Swarm (`swarm_chat`)**:
   Query 7 frontier models unified under high-availability failover aliases:
   - `deepseek-v4-pro`: Deep reasoning and algorithmic problem solving (Primary: SiliconFlow DeepSeek-V3/V4 -> OpenRouter DeepSeek-R1 -> Groq Llama-3.3).
   - `kimi-k3`: Long-context and technical document analysis (Primary: Moonshot v1-32k -> SiliconFlow DeepSeek-V3 -> Zhipu GLM-4-Flash).
   - `glm-5.3`: Multi-perspective auditing and specification compliance (Primary: Zhipu GLM-4-Flash -> SiliconFlow GLM-9B -> OpenRouter Llama-3.3).
   - `qwen-3.8`: Precision code synthesis and syntax patching (Primary: DashScope Qwen-Plus -> SiliconFlow Qwen2.5-7B -> OpenRouter Qwen-2.5-Coder-32B).
   - `gemini-3.8-flash-high`: Architectural planning and outer milestone progression (Primary: Google Gemini 2.5 Flash -> OpenRouter Gemini 2.0 -> Groq Llama-3.3).
   - `muse-spark`: Creative synthesis and rapid ideation (Primary: OpenRouter Llama-3.3 -> Cerebras Llama-3.1-70b -> SiliconFlow Qwen2.5).

2. **3-Tier Autonomous Feedback Loops (`swarm_run_autonomous_loop`)**:
   - **Tier 1 (Inner Local Test Loop)**: Subprocess pytest runner with automated traceback extraction and iterative self-repair with a 5-attempt budget.
   - **Tier 2 (Middle Adversarial Audit Loop)**: GLM-5.3 Judge evaluates spec compliance, security isolation, and robustness before changes land.
   - **Tier 3 (Outer Goal Convergence Loop)**: Gemini Chief Architect updates the milestone state machine and persists state atomically to `checkpoint.json`.

3. **Inline Context Compression Engine (`swarm_compress_context`)**:
   - SHA-256 system prompt deduplication.
   - Conversational pleasantry and reasoning tag pruning.
   - Sliding-window compaction preserving code fences and invariants.
   - Typically yields 1.3x to 3.0x token throughput multipliers.

4. **Guarded Host System Execution (`swarm_powershell_exec`)**:
   - UTF-8 preamble with elevated admin privilege detection.
   - Process tree termination on timeout (`taskkill /F /T`).

5. **Telemetry & Model Routing Catalog (`swarm_system_status`, `swarm_model_catalog`)**:
   - Inspect live circuit breakers, rate limits, cooldown timers, and tokens saved.

---

## When to Use Swarm Tools

- **When complex reasoning is needed**: Delegate prompts to `swarm_chat(model="deepseek-v4-pro", prompt=...)`.
- **When writing or fixing code against tests**: Invoke `swarm_run_autonomous_loop(task_description=..., file_path=...)`.
- **When you need independent peer review**: Prompt GLM-5.3 or Kimi-K3 via `swarm_chat(model="glm-5.3", prompt="Audit this implementation: ...")`.
- **When token context is large**: Use `swarm_compress_context(messages=...)` to compact history before calling frontier models.

---

## Temporary Integration Control

This integration is completely modular and temporary:
- **Attach**: Run `attach_to_antigravity.bat` from `C:\Users\Admin\.gemini\antigravity\scratch\swarm-agent`.
- **Detach**: Run `detach_from_antigravity.bat` from `C:\Users\Admin\.gemini\antigravity\scratch\swarm-agent`. Reverts all configuration files byte-for-byte and leaves zero residue.
