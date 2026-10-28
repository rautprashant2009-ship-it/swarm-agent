"""Interactive Setup Wizard for Swarm Agent Token Gateway.

Collects, validates, verifies, and persists API keys for 8 frontier free-tier providers:
- Google AI Studio (Gemini)
- SiliconFlow (DeepSeek)
- Moonshot AI (Kimi)
- Zhipu BigModel (GLM)
- Alibaba DashScope (Qwen)
- OpenRouter (Unified router / Free pool)
- Groq (Llama-3.3 ultra-fast inference)
- Cerebras (Llama-3.1 wafer-scale engine)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

CONFIG_PATH = Path("config.json")
EXAMPLE_CONFIG_PATH = Path("config.example.json")

PROVIDERS: List[Dict[str, Any]] = [
    {
        "id": "google",
        "name": "Google AI Studio (Gemini)",
        "env_var": "GEMINI_API_KEY",
        "alt_env_var": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "pattern": r"^AIzaSy[A-Za-z0-9_-]{33}$",
        "test_endpoint": "models",
        "default_model": "gemini-3.8-flash-high",
        "help_url": "https://aistudio.google.com/app/apikey",
    },
    {
        "id": "siliconflow",
        "name": "SiliconFlow (DeepSeek-V4/Pro)",
        "env_var": "SILICONFLOW_API_KEY",
        "alt_env_var": None,
        "base_url": "https://api.siliconflow.cn/v1",
        "pattern": r"^sk-[a-zA-Z0-9]{32,64}$",
        "test_endpoint": "models",
        "default_model": "deepseek-ai/DeepSeek-V4",
        "help_url": "https://cloud.siliconflow.cn/account/ak",
    },
    {
        "id": "moonshot",
        "name": "Moonshot AI (Kimi-K3)",
        "env_var": "MOONSHOT_API_KEY",
        "alt_env_var": None,
        "base_url": "https://api.moonshot.cn/v1",
        "pattern": r"^sk-[a-zA-Z0-9]{40,64}$",
        "test_endpoint": "models",
        "default_model": "moonshot-v1-8k",
        "help_url": "https://platform.moonshot.cn/console/api-keys",
    },
    {
        "id": "zhipu",
        "name": "Zhipu BigModel (GLM-5.3)",
        "env_var": "ZHIPU_API_KEY",
        "alt_env_var": None,
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "pattern": r"^([a-zA-Z0-9]+\.[a-zA-Z0-9]+|sk-[a-zA-Z0-9]{32,64})$",
        "test_endpoint": "models",
        "default_model": "glm-4-flash",
        "help_url": "https://bigmodel.cn/usercenter/apikeys",
    },
    {
        "id": "dashscope",
        "name": "Alibaba DashScope (Qwen 3.8)",
        "env_var": "DASHSCOPE_API_KEY",
        "alt_env_var": None,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "pattern": r"^sk-[a-zA-Z0-9]{32,48}$",
        "test_endpoint": "models",
        "default_model": "qwen-turbo",
        "help_url": "https://dashscope.console.aliyun.com/apiKey",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter (Free Pool & Unified)",
        "env_var": "OPENROUTER_API_KEY",
        "alt_env_var": None,
        "base_url": "https://openrouter.ai/api/v1",
        "pattern": r"^sk-or-v1-[a-f0-9]{64}$",
        "test_endpoint": "models",
        "default_model": "openrouter/auto",
        "help_url": "https://openrouter.ai/keys",
    },
    {
        "id": "groq",
        "name": "Groq (Llama-3.3 Ultra-Fast)",
        "env_var": "GROQ_API_KEY",
        "alt_env_var": None,
        "base_url": "https://api.groq.com/openai/v1",
        "pattern": r"^gsk_[a-zA-Z0-9]{48,64}$",
        "test_endpoint": "models",
        "default_model": "llama-3.3-70b-versatile",
        "help_url": "https://console.groq.com/keys",
    },
    {
        "id": "cerebras",
        "name": "Cerebras (Llama-3.1 Wafer-Scale)",
        "env_var": "CEREBRAS_API_KEY",
        "alt_env_var": None,
        "base_url": "https://api.cerebras.ai/v1",
        "pattern": r"^csk-[a-zA-Z0-9]{48,64}$",
        "test_endpoint": "models",
        "default_model": "llama3.1-8b",
        "help_url": "https://cloud.cerebras.ai/",
    },
]


def validate_key_format(key: str, pattern: str) -> bool:
    """Validate whether an API key matches expected provider pattern."""
    if not key or not key.strip():
        return True  # Blank key is valid (means provider not configured / skipped)
    return bool(re.match(pattern, key.strip()))


def test_connectivity(
    base_url: str,
    api_key: str,
    endpoint: str = "models",
    timeout: float = 6.0,
) -> Tuple[bool, str]:
    """Test live connectivity against the provider endpoint."""
    if not api_key or not api_key.strip():
        return False, "Skipped (no key provided)"

    clean_base = base_url.rstrip("/")
    clean_endpoint = endpoint.lstrip("/")
    url = f"{clean_base}/{clean_endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "Swarm-Agent-SetupWizard/1.0",
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code in (200, 201):
                return True, f"Online (HTTP {resp.status_code})"
            elif resp.status_code == 429:
                return True, "Online - Rate Limited (HTTP 429: Key valid)"
            elif resp.status_code in (401, 403):
                return False, f"Auth Error (HTTP {resp.status_code}: Invalid key)"
            else:
                return False, f"Unexpected response (HTTP {resp.status_code})"
    except httpx.ConnectTimeout:
        return False, "Connection timed out"
    except httpx.ConnectError:
        return False, "Endpoint unreachable (DNS/Network error)"
    except Exception as exc:
        return False, f"Verification failed: {exc}"


def load_template_config(base_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load base configuration from config.json or config.example.json."""
    paths_to_try = [
        base_path,
        CONFIG_PATH,
        EXAMPLE_CONFIG_PATH,
    ]
    for p in paths_to_try:
        if p and Path(p).is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    # Minimal fallback structure if no template file exists
    return {
        "server": {"host": "127.0.0.1", "port": 8000, "cors_origins": ["*"]},
        "providers": {},
        "alias_mappings": {},
        "compression": {"enabled": True, "history_window_turns": 10},
        "tools": {"powershell_enabled": True, "powershell_timeout_seconds": 60},
    }


def build_config_dict(
    host: str,
    port: int,
    provider_keys: Dict[str, str],
    existing_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete configuration dictionary merging keys and existing settings."""
    cfg = dict(existing_config or load_template_config())

    if "server" not in cfg:
        cfg["server"] = {}
    cfg["server"]["host"] = host
    cfg["server"]["port"] = port
    if "cors_origins" not in cfg["server"]:
        cfg["server"]["cors_origins"] = ["*"]

    if "providers" not in cfg:
        cfg["providers"] = {}

    for prov in PROVIDERS:
        pid = prov["id"]
        key = provider_keys.get(pid, "").strip()
        current_prov = cfg["providers"].get(pid, {})

        current_prov["base_url"] = current_prov.get("base_url", prov["base_url"])
        current_prov["enabled"] = bool(key) or current_prov.get("enabled", True)
        current_prov["api_key"] = key
        if "priority" not in current_prov:
            current_prov["priority"] = 1 if pid in ["google", "siliconflow", "moonshot", "zhipu", "dashscope"] else 2
        if "base_cooldown_seconds" not in current_prov:
            current_prov["base_cooldown_seconds"] = 30.0

        cfg["providers"][pid] = current_prov

    return cfg


def save_config(config_dict: Dict[str, Any], target_path: Path = CONFIG_PATH) -> Path:
    """Save configuration atomically to disk."""
    target_path = Path(target_path)
    tmp_path = target_path.with_suffix(".tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")

    os.replace(tmp_path, target_path)
    return target_path


def run_wizard(
    interactive: bool = True,
    target_path: str = "config.json",
    host: str = "127.0.0.1",
    port: int = 8000,
    keys: Optional[Dict[str, str]] = None,
    check_connectivity: bool = True,
) -> Dict[str, Any]:
    """Execute the setup wizard, supporting interactive and headless modes."""
    console = Console() if HAS_RICH else None
    existing_cfg = load_template_config()
    existing_providers = existing_cfg.get("providers", {})
    existing_server = existing_cfg.get("server", {})

    resolved_host = host or existing_server.get("host", "127.0.0.1")
    resolved_port = port or existing_server.get("port", 8000)
    provider_keys: Dict[str, str] = {}
    connectivity_results: Dict[str, Tuple[bool, str]] = {}

    if console and interactive:
        header_text = Text(
            "Swarm Agent Gateway - Interactive Setup Wizard\n"
            "Aggregating 8 Free-Tier Frontier Providers into a Unified OpenAI Proxy",
            style="bold cyan",
            justify="center",
        )
        console.print(Panel(header_text, border_style="bright_blue"))
        console.print(
            "[dim]Press Enter to accept defaults. Keys can be set now or left blank to use environment variables.[/dim]\n"
        )

    # 1. Server settings
    if interactive and console:
        resolved_host = Prompt.ask("[bold green]Gateway Server Host[/bold green]", default=resolved_host)
        resolved_port = int(
            Prompt.ask("[bold green]Gateway Server Port[/bold green]", default=str(resolved_port))
        )
    elif interactive and not console:
        inp_host = input(f"Gateway Server Host [{resolved_host}]: ").strip()
        if inp_host:
            resolved_host = inp_host
        inp_port = input(f"Gateway Server Port [{resolved_port}]: ").strip()
        if inp_port and inp_port.isdigit():
            resolved_port = int(inp_port)

    # 2. Key collection and verification for 8 providers
    for prov in PROVIDERS:
        pid = prov["id"]
        name = prov["name"]
        env_var = prov["env_var"]
        alt_env_var = prov.get("alt_env_var")
        pattern = prov["pattern"]
        base_url = prov["base_url"]
        help_url = prov["help_url"]

        # Determine default key from parameters, env, or existing config
        current_key = ""
        if keys and pid in keys:
            current_key = keys[pid]
        elif os.environ.get(env_var):
            current_key = os.environ[env_var]
        elif alt_env_var and os.environ.get(alt_env_var):
            current_key = os.environ[alt_env_var]
        elif pid in existing_providers:
            existing_val = existing_providers[pid].get("api_key", "")
            if existing_val and not existing_val.startswith("${"):
                current_key = existing_val

        if interactive:
            display_default = f"...{current_key[-6:]}" if len(current_key) > 8 else (current_key or "None")
            prompt_label = (
                f"[bold cyan]{name}[/bold cyan] Key\n"
                f"  [dim]Env: {env_var} | Get key: {help_url}[/dim]\n"
                f"  Current: {display_default}"
            )

            while True:
                if console:
                    entered = Prompt.ask(prompt_label, default=current_key, password=True)
                else:
                    entered = input(f"{name} Key [{display_default}]: ").strip()
                    if not entered:
                        entered = current_key

                entered = entered.strip()
                if validate_key_format(entered, pattern):
                    current_key = entered
                    break
                else:
                    msg = f"[bold yellow]Warning:[/bold yellow] Key does not match expected pattern ({pattern})."
                    if console:
                        console.print(msg)
                        if Confirm.ask("Use this key anyway?", default=True):
                            current_key = entered
                            break
                    else:
                        print(f"Warning: Key does not match expected pattern {pattern}.")
                        current_key = entered
                        break
        else:
            # Headless / automated mode
            if keys and pid in keys:
                current_key = keys[pid].strip()

        provider_keys[pid] = current_key

        # Live connectivity check
        if check_connectivity and current_key:
            if console and interactive:
                with console.status(f"[cyan]Verifying connection to {name}...[/cyan]"):
                    ok, detail = test_connectivity(base_url, current_key, prov["test_endpoint"])
            else:
                ok, detail = test_connectivity(base_url, current_key, prov["test_endpoint"])
            connectivity_results[pid] = (ok, detail)
        else:
            connectivity_results[pid] = (False, "Skipped" if not current_key else "Not checked")

    # 3. Build & Save Config
    final_config = build_config_dict(
        host=resolved_host,
        port=resolved_port,
        provider_keys=provider_keys,
        existing_config=existing_cfg,
    )
    saved_path = save_config(final_config, Path(target_path))

    # 4. Summary table
    if console and interactive:
        table = Table(title="Swarm Gateway Provider Verification Summary", border_style="blue")
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Key Status", style="magenta")
        table.add_column("Connectivity", style="green")
        table.add_column("Endpoint", style="dim")

        for prov in PROVIDERS:
            pid = prov["id"]
            k = provider_keys.get(pid, "")
            ok, detail = connectivity_results.get(pid, (False, "N/A"))
            key_status = "[green]Configured[/green]" if k else "[yellow]Unset (Env fallback)[/yellow]"
            conn_status = f"[green]{detail}[/green]" if ok else f"[yellow]{detail}[/yellow]"
            table.add_row(prov["name"], key_status, conn_status, prov["base_url"])

        console.print("\n")
        console.print(table)
        console.print(f"\n[bold green]✓ Configuration saved successfully to {saved_path.resolve()}[/bold green]")
        console.print(f"[dim]Run 'python -m swarm.main' or 'run_swarm.bat' to start the gateway.[/dim]\n")
    elif not console and interactive:
        print(f"\nConfiguration saved successfully to {saved_path.resolve()}\n")

    return final_config


def main() -> None:
    """CLI entrypoint for python -m swarm.cli.wizard."""
    parser = argparse.ArgumentParser(description="Swarm Agent Setup Wizard")
    parser.add_argument("--target", default="config.json", help="Path to output config.json")
    parser.add_argument("--host", default="127.0.0.1", help="Gateway server host")
    parser.add_argument("--port", type=int, default=8000, help="Gateway server port")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--skip-connectivity", action="store_true", help="Skip live endpoint network checks")
    parser.add_argument("--keys", type=str, default=None, help="JSON string of provider keys mapping")

    args = parser.parse_args()

    parsed_keys = None
    if args.keys:
        try:
            parsed_keys = json.loads(args.keys)
        except Exception as e:
            print(f"Error parsing --keys JSON: {e}", file=sys.stderr)
            sys.exit(1)

    run_wizard(
        interactive=not args.non_interactive,
        target_path=args.target,
        host=args.host,
        port=args.port,
        keys=parsed_keys,
        check_connectivity=not args.skip_connectivity,
    )


if __name__ == "__main__":
    main()
