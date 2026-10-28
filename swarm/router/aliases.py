"""Frontier model alias and role resolution engine."""

from __future__ import annotations
from typing import Dict, List, NamedTuple, Optional


class CandidateTarget(NamedTuple):
    provider: str
    model: str


ROLE_TO_ALIAS: Dict[str, str] = {
    "swarm-architect": "gemini-3.8-flash-high",
    "swarm-coder": "qwen-3.8",
    "swarm-reasoner": "deepseek-v4-pro",
    "swarm-judge": "glm-5.3",
}

DEFAULT_ALIAS_MAP: Dict[str, List[CandidateTarget]] = {
    "deepseek-v4.1-flash-1017": [
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V4.1-Flash-1017"),
        CandidateTarget("openrouter", "deepseek/deepseek-flash"),
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V3"),
    ],
    "deepseek-v4.1-flash": [
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V4.1-Flash-1017"),
        CandidateTarget("openrouter", "deepseek/deepseek-flash"),
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V3"),
    ],
    "deepseek-v4-pro-0813": [
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V4-Pro-0813"),
        CandidateTarget("openrouter", "deepseek/deepseek-v4-pro-0813"),
        CandidateTarget("openrouter", "deepseek/deepseek-r1:free"),
    ],
    "deepseek-v4-pro": [
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V4-Pro-0813"),
        CandidateTarget("openrouter", "deepseek/deepseek-v4-pro-0813"),
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V3"),
    ],
    "deepseek-v4": [
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V4-Pro-0813"),
        CandidateTarget("openrouter", "deepseek/deepseek-flash"),
        CandidateTarget("siliconflow", "deepseek-ai/DeepSeek-V3"),
    ],
    "kimi-k3-swarm": [
        CandidateTarget("moonshot", "K3-Swarm-Max"),
        CandidateTarget("moonshot", "moonshotai/Kimi-K3"),
        CandidateTarget("moonshot", "moonshot-v1-32k"),
    ],
    "kimi-k3": [
        CandidateTarget("moonshot", "K3-Swarm-Max"),
        CandidateTarget("moonshot", "moonshotai/Kimi-K3"),
        CandidateTarget("moonshot", "moonshot-v1-32k"),
    ],
    "glm-5.3": [
        CandidateTarget("zhipu", "glm-5.3"),
        CandidateTarget("siliconflow", "zai-org/GLM-5.3"),
        CandidateTarget("zhipu", "glm-4-flash"),
    ],
    "glm-5.3-max": [
        CandidateTarget("zhipu", "glm-5.3"),
        CandidateTarget("siliconflow", "zai-org/GLM-5.3"),
        CandidateTarget("zhipu", "glm-4-flash"),
    ],
    "glm-4.5": [
        CandidateTarget("zhipu", "glm-4-flash"),
        CandidateTarget("siliconflow", "THUDM/glm-4-9b-chat"),
        CandidateTarget("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
    ],
    "qwen-3.8-max": [
        CandidateTarget("dashscope", "qwen3.8-max"),
        CandidateTarget("openrouter", "qwen/qwen3.8-max"),
        CandidateTarget("siliconflow", "Qwen/Qwen3.8-2.4T-A95B"),
    ],
    "qwen-3.8": [
        CandidateTarget("dashscope", "qwen3.8-max"),
        CandidateTarget("openrouter", "qwen/qwen3.8-max"),
        CandidateTarget("dashscope", "qwen-plus"),
    ],
    "qwen-2.5-coder": [
        CandidateTarget("dashscope", "qwen-plus"),
        CandidateTarget("siliconflow", "Qwen/Qwen2.5-7B-Instruct"),
        CandidateTarget("openrouter", "qwen/qwen-2.5-coder-32b-instruct:free"),
    ],
    "gemini-3.8-flash-high": [
        CandidateTarget("google", "gemini-3.8-flash"),
        CandidateTarget("google", "gemini-2.5-flash"),
        CandidateTarget("openrouter", "google/gemini-2.0-flash-exp:free"),
    ],
    "gemini-3.8-flash": [
        CandidateTarget("google", "gemini-3.8-flash"),
        CandidateTarget("google", "gemini-2.5-flash"),
        CandidateTarget("openrouter", "google/gemini-2.0-flash-exp:free"),
    ],
    "gemini-flash": [
        CandidateTarget("google", "gemini-3.8-flash"),
        CandidateTarget("google", "gemini-2.5-flash"),
        CandidateTarget("openrouter", "google/gemini-2.0-flash-exp:free"),
    ],
    "muse-spark-open": [
        CandidateTarget("openrouter", "meta-models/Muse-Spark-Open-816B-A42B"),
        CandidateTarget("cerebras", "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"),
        CandidateTarget("cerebras", "llama3.1-70b"),
    ],
    "muse-spark": [
        CandidateTarget("openrouter", "meta-models/Muse-Spark-Open-816B-A42B"),
        CandidateTarget("cerebras", "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"),
        CandidateTarget("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
    ],
}

KNOWN_PROVIDER_MODELS: Dict[str, str] = {
    # Google
    "gemini-3.8-flash": "google",
    "gemini-2.5-flash": "google",
    "gemini-2.5-pro": "google",
    # DeepSeek
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": "siliconflow",
    "deepseek-ai/DeepSeek-V4-Pro-0813": "siliconflow",
    "deepseek-flash": "siliconflow",
    "deepseek-v4-pro-0813": "openrouter",
    "deepseek-ai/DeepSeek-V3": "siliconflow",
    "deepseek-ai/DeepSeek-R1": "siliconflow",
    # Moonshot
    "K3-Swarm-Max": "moonshot",
    "moonshotai/Kimi-K3": "moonshot",
    "moonshot-v1-8k": "moonshot",
    "moonshot-v1-32k": "moonshot",
    "moonshot-v1-128k": "moonshot",
    # Zhipu
    "glm-5.3": "zhipu",
    "zai-org/GLM-5.3": "siliconflow",
    "glm-4-flash": "zhipu",
    # DashScope / Alibaba
    "qwen3.8-max": "dashscope",
    "Qwen/Qwen3.8-2.4T-A95B": "siliconflow",
    "qwen-plus": "dashscope",
    "qwen-turbo": "dashscope",
    "qwen-max": "dashscope",
    "qwen2.5-72b-instruct": "dashscope",
    # Meta / Muse Spark
    "meta-models/Muse-Spark-Open-816B-A42B": "openrouter",
    "meta-models/Muse-Spark-Open-816B-A42B-NVFP4": "cerebras",
    # Groq & Cerebras
    "llama-3.3-70b-versatile": "groq",
    "llama-3.1-8b-instant": "groq",
    "mixtral-8x7b-32768": "groq",
    "llama3.1-8b": "cerebras",
    "llama3.1-70b": "cerebras",
}


class AliasResolver:
    """Dynamic resolution of model aliases and swarm roles to ordered candidate targets."""

    def __init__(self, custom_mappings: Optional[Dict[str, List[CandidateTarget]]] = None):
        self.mappings: Dict[str, List[CandidateTarget]] = dict(DEFAULT_ALIAS_MAP)
        if custom_mappings:
            self.mappings.update(custom_mappings)

    def resolve(self, model_or_role: str) -> List[CandidateTarget]:
        """Resolve a requested model, alias, or swarm role to an ordered candidate list."""
        if not model_or_role:
            return []

        clean_name = model_or_role.strip()
        lower_name = clean_name.lower()

        # 1. Swarm role check
        if lower_name in ROLE_TO_ALIAS:
            target_alias = ROLE_TO_ALIAS[lower_name]
            return self.mappings.get(target_alias, [])

        # 2. Known alias check
        if lower_name in self.mappings:
            return self.mappings[lower_name]

        # 3. Known direct provider model match
        if clean_name in KNOWN_PROVIDER_MODELS:
            provider = KNOWN_PROVIDER_MODELS[clean_name]
            return [CandidateTarget(provider, clean_name)]

        # 4. Pattern-based passthrough matching
        if clean_name.startswith("gemini-"):
            return [CandidateTarget("google", clean_name)]
        if clean_name.startswith("moonshot-"):
            return [CandidateTarget("moonshot", clean_name)]
        if clean_name.startswith("glm-"):
            return [CandidateTarget("zhipu", clean_name)]
        if clean_name.startswith("qwen-") or clean_name.startswith("qwen2."):
            return [
                CandidateTarget("dashscope", clean_name),
                CandidateTarget("siliconflow", clean_name),
            ]
        if clean_name.endswith(":free"):
            return [CandidateTarget("openrouter", clean_name)]
        if "/" in clean_name:
            # e.g., org/model format often supported by SiliconFlow or OpenRouter
            return [
                CandidateTarget("siliconflow", clean_name),
                CandidateTarget("openrouter", clean_name),
            ]
        if "llama" in lower_name or "mixtral" in lower_name:
            return [
                CandidateTarget("groq", clean_name),
                CandidateTarget("cerebras", clean_name),
            ]

        # 5. Fallback passthrough to openrouter or all configured providers
        return [
            CandidateTarget("openrouter", clean_name),
            CandidateTarget("siliconflow", clean_name),
        ]

    def get_all_aliases(self) -> List[str]:
        """Return list of supported aliases and roles."""
        aliases = list(self.mappings.keys())
        roles = list(ROLE_TO_ALIAS.keys())
        return sorted(list(set(aliases + roles)))

    def get_catalog(self) -> List[str]:
        """Return full catalog of virtual aliases and known provider models."""
        return sorted(list(set(self.get_all_aliases() + list(KNOWN_PROVIDER_MODELS.keys()))))
