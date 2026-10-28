"""Configuration loader for the Swarm Agent Gateway.
Supports dual-source configuration: config.json file with environment variable overrides.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class ProviderConfig(BaseModel):
    api_key: str = ""
    base_url: str
    enabled: bool = True
    priority: int = 1
    base_cooldown_seconds: float = 30.0


class TargetCandidate(BaseModel):
    provider: str
    model: str


DEFAULT_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "google": {
        "api_key": "",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "enabled": True,
        "priority": 1,
        "base_cooldown_seconds": 30.0,
    },
    "siliconflow": {
        "api_key": "",
        "base_url": "https://api.siliconflow.cn/v1",
        "enabled": True,
        "priority": 1,
        "base_cooldown_seconds": 30.0,
    },
    "moonshot": {
        "api_key": "",
        "base_url": "https://api.moonshot.cn/v1",
        "enabled": True,
        "priority": 1,
        "base_cooldown_seconds": 60.0,
    },
    "zhipu": {
        "api_key": "",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "enabled": True,
        "priority": 1,
        "base_cooldown_seconds": 30.0,
    },
    "dashscope": {
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "enabled": True,
        "priority": 1,
        "base_cooldown_seconds": 30.0,
    },
    "openrouter": {
        "api_key": "",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": True,
        "priority": 2,
        "base_cooldown_seconds": 60.0,
    },
    "groq": {
        "api_key": "",
        "base_url": "https://api.groq.com/openai/v1",
        "enabled": True,
        "priority": 2,
        "base_cooldown_seconds": 30.0,
    },
    "cerebras": {
        "api_key": "",
        "base_url": "https://api.cerebras.ai/v1",
        "enabled": True,
        "priority": 2,
        "base_cooldown_seconds": 30.0,
    },
}

DEFAULT_ALIAS_MAPPINGS: Dict[str, List[Dict[str, str]]] = {
    "deepseek-v4-pro": [
        {"provider": "siliconflow", "model": "deepseek-ai/DeepSeek-V3"},
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    ],
    "deepseek-v4": [
        {"provider": "siliconflow", "model": "deepseek-ai/DeepSeek-V3"},
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    ],
    "kimi-k3": [
        {"provider": "moonshot", "model": "moonshot-v1-32k"},
        {"provider": "siliconflow", "model": "deepseek-ai/DeepSeek-V3"},
        {"provider": "zhipu", "model": "glm-4-flash"},
    ],
    "glm-5.3": [
        {"provider": "zhipu", "model": "glm-4-flash"},
        {"provider": "siliconflow", "model": "THUDM/glm-4-9b-chat"},
        {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"},
    ],
    "glm-4.5": [
        {"provider": "zhipu", "model": "glm-4-flash"},
        {"provider": "siliconflow", "model": "THUDM/glm-4-9b-chat"},
        {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"},
    ],
    "qwen-3.8": [
        {"provider": "dashscope", "model": "qwen-plus"},
        {"provider": "siliconflow", "model": "Qwen/Qwen2.5-7B-Instruct"},
        {"provider": "openrouter", "model": "qwen/qwen-2.5-coder-32b-instruct:free"},
    ],
    "qwen-2.5-coder": [
        {"provider": "dashscope", "model": "qwen-plus"},
        {"provider": "siliconflow", "model": "Qwen/Qwen2.5-7B-Instruct"},
        {"provider": "openrouter", "model": "qwen/qwen-2.5-coder-32b-instruct:free"},
    ],
    "gemini-3.8-flash-high": [
        {"provider": "google", "model": "gemini-2.5-flash"},
        {"provider": "openrouter", "model": "google/gemini-2.0-flash-exp:free"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    ],
    "gemini-flash": [
        {"provider": "google", "model": "gemini-2.5-flash"},
        {"provider": "openrouter", "model": "google/gemini-2.0-flash-exp:free"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    ],
    "muse-spark": [
        {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"},
        {"provider": "cerebras", "model": "llama3.1-70b"},
        {"provider": "siliconflow", "model": "Qwen/Qwen2.5-7B-Instruct"},
    ],
}

ENV_KEY_MAP = {
    "google": "GEMINI_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}


class GatewayConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    alias_mappings: Dict[str, List[TargetCandidate]] = Field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> GatewayConfig:
    """Load gateway configuration from JSON file and environment variables."""
    raw_data: Dict[str, Any] = {}

    # 1. Search for config file
    candidate_paths = [
        config_path,
        os.environ.get("SWARM_CONFIG_PATH"),
        "config.json",
        "config.example.json",
    ]
    for p in candidate_paths:
        if p and Path(p).is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    break
            except Exception:
                pass

    # 2. Server config resolution
    server_data = raw_data.get("server", {})
    host = os.environ.get("SWARM_GATEWAY_HOST", server_data.get("host", "127.0.0.1"))
    port_str = os.environ.get("SWARM_GATEWAY_PORT")
    port = int(port_str) if port_str else int(server_data.get("port", 8000))
    cors_origins = server_data.get("cors_origins", ["*"])
    server_cfg = ServerConfig(host=host, port=port, cors_origins=cors_origins)

    # 3. Providers config resolution
    providers_cfg: Dict[str, ProviderConfig] = {}
    file_providers = raw_data.get("providers", {})
    for p_name, default_dict in DEFAULT_PROVIDERS.items():
        file_dict = file_providers.get(p_name, {})
        base_url = file_dict.get("base_url", default_dict["base_url"])
        enabled = file_dict.get("enabled", default_dict["enabled"])
        priority = file_dict.get("priority", default_dict["priority"])
        base_cooldown = float(file_dict.get("base_cooldown_seconds", default_dict["base_cooldown_seconds"]))

        # Environment variable override takes precedence
        env_var_name = ENV_KEY_MAP.get(p_name)
        env_api_key = os.environ.get(env_var_name, "") if env_var_name else ""
        api_key = env_api_key or file_dict.get("api_key", default_dict["api_key"])

        providers_cfg[p_name] = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            enabled=enabled,
            priority=priority,
            base_cooldown_seconds=base_cooldown,
        )

    # 4. Alias mappings resolution
    alias_data = raw_data.get("alias_mappings", DEFAULT_ALIAS_MAPPINGS)
    resolved_aliases: Dict[str, List[TargetCandidate]] = {}
    for alias, targets in alias_data.items():
        resolved_aliases[alias] = [
            TargetCandidate(provider=t["provider"], model=t["model"]) for t in targets
        ]

    # Ensure all default aliases exist
    for alias, targets in DEFAULT_ALIAS_MAPPINGS.items():
        if alias not in resolved_aliases:
            resolved_aliases[alias] = [
                TargetCandidate(provider=t["provider"], model=t["model"]) for t in targets
            ]

    return GatewayConfig(
        server=server_cfg,
        providers=providers_cfg,
        alias_mappings=resolved_aliases,
    )


_GLOBAL_CONFIG: Optional[GatewayConfig] = None


def get_config(reload: bool = False, config_path: Optional[str] = None) -> GatewayConfig:
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        _GLOBAL_CONFIG = load_config(config_path)
    return _GLOBAL_CONFIG
