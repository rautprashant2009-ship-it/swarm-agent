"""Standalone Command Line Interface for Swarm Agent Gateway & Swarm Runtime.

Commands:
  start     - Start the multi-provider gateway ASGI server (uvicorn)
  wizard    - Launch interactive configuration wizard for 8 free-tier providers
  status    - Check health and circuit breaker status of running gateway
  models    - List available models, frontier aliases, and fallback chains
  test      - Execute automated test suite (pytest) with coverage and reporting
  compress  - Test context compression engine on prompt files or sample turns
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

import click
import httpx

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from swarm.config import get_config
from swarm.cli.wizard import run_wizard


@click.group(name="swarm", help="Swarm Agent Multi-Provider Gateway & Autonomous Runtime CLI.")
@click.version_option(version="1.0.0", prog_name="swarm")
def cli():
    """Swarm Agent CLI entrypoint."""
    pass


@cli.command("start", help="Start the Swarm multi-provider gateway server.")
@click.option("--host", "-h", default=None, help="Host interface to bind (default from config or 127.0.0.1)")
@click.option("--port", "-p", default=None, type=int, help="Port to bind (default from config or 8000)")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload for development")
@click.option("--config-path", "-c", default=None, help="Path to config.json file")
def start_server(host: Optional[str], port: Optional[int], reload: bool, config_path: Optional[str]):
    """Launch the ASGI gateway server via Uvicorn."""
    console = Console() if HAS_RICH else None
    
    if config_path:
        os.environ["SWARM_CONFIG_PATH"] = config_path

    cfg = get_config(reload=True, config_path=config_path)
    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port

    if console:
        console.print(
            Panel(
                Text(
                    f"Starting Swarm Multi-Provider Gateway\n"
                    f"Listening on: http://{bind_host}:{bind_port}\n"
                    f"OpenAI Endpoint: http://{bind_host}:{bind_port}/v1/chat/completions\n"
                    f"Health Endpoint: http://{bind_host}:{bind_port}/health",
                    style="bold green",
                ),
                title="Swarm Gateway Daemon",
                border_style="bright_blue",
            )
        )
    else:
        print(f"[*] Starting Swarm Gateway on http://{bind_host}:{bind_port} ...")

    import uvicorn
    uvicorn.run("swarm.main:app", host=bind_host, port=bind_port, reload=reload)


@cli.command("wizard", help="Run the interactive setup wizard for API keys and configuration.")
@click.option("--target", "-t", default="config.json", help="Destination path for configuration file")
@click.option("--host", default="127.0.0.1", help="Default server host")
@click.option("--port", default=8000, type=int, help="Default server port")
@click.option("--non-interactive", is_flag=True, default=False, help="Run without user interaction")
@click.option("--skip-connectivity", is_flag=True, default=False, help="Skip live endpoint connectivity verification")
def wizard(target: str, host: str, port: int, non_interactive: bool, skip_connectivity: bool):
    """Launch the interactive setup wizard."""
    run_wizard(
        interactive=not non_interactive,
        target_path=target,
        host=host,
        port=port,
        check_connectivity=not skip_connectivity,
    )


@cli.command("status", help="Inspect health and circuit breaker status of local gateway.")
@click.option("--host", default=None, help="Gateway host (default: from config or 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Gateway port (default: from config or 8000)")
def check_status(host: Optional[str], port: Optional[int]):
    """Query /health endpoint and display live provider circuit breaker state."""
    console = Console() if HAS_RICH else None
    cfg = get_config()
    target_host = host or cfg.server.host
    target_port = port or cfg.server.port
    health_url = f"http://{target_host}:{target_port}/health"

    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(health_url)
            if resp.status_code != 200:
                msg = f"Gateway returned non-200 status code: {resp.status_code}"
                if console:
                    console.print(f"[bold red]{msg}[/bold red]")
                else:
                    print(msg)
                sys.exit(1)
            data = resp.json()
    except httpx.ConnectError:
        err_msg = (
            f"[bold red]Connection Refused:[/bold red] Gateway is not running at {health_url}.\n"
            f"Run [bold cyan]python -m swarm start[/bold cyan] or [bold cyan]run_swarm.bat[/bold cyan] to launch."
        )
        if console:
            console.print(err_msg)
        else:
            print(f"Error: Gateway is not running at {health_url}. Run 'swarm start' to launch.")
        sys.exit(1)
    except Exception as exc:
        if console:
            console.print(f"[bold red]Error checking status:[/bold red] {exc}")
        else:
            print(f"Error checking status: {exc}")
        sys.exit(1)

    if console:
        overall_status = data.get("status", "unknown")
        status_color = "green" if overall_status == "healthy" else "yellow"
        console.print(f"\n[bold]Gateway Status:[/bold] [{status_color}]{overall_status.upper()}[/{status_color}]")
        console.print(f"[dim]Version: {data.get('version', '1.0.0')} | Providers: {data.get('active_providers_count', 0)}[/dim]\n")

        table = Table(title="Provider Circuit Breakers & Pool State", border_style="blue")
        table.add_column("Provider", style="cyan")
        table.add_column("State", style="magenta")
        table.add_column("Failures", justify="right")
        table.add_column("Cooldown Remaining", justify="right")

        breakers = data.get("circuit_breakers", {})
        for name, details in breakers.items():
            state = details.get("state", "UNKNOWN")
            color = "green" if state == "HEALTHY" else ("yellow" if state == "HALF_OPEN" else "red")
            fails = str(details.get("failure_count", 0))
            cooldown = f"{details.get('cooldown_remaining_seconds', 0.0):.1f}s"
            table.add_row(name, f"[{color}]{state}[/{color}]", fails, cooldown)

        console.print(table)
    else:
        print(f"Gateway Status: {data.get('status', 'unknown')}")
        print(f"Circuit Breakers: {json.dumps(data.get('circuit_breakers', {}), indent=2)}")


@cli.command("models", help="List available models and frontier alias fallback chains.")
@click.option("--host", default=None, help="Gateway host")
@click.option("--port", default=None, type=int, help="Gateway port")
def list_models(host: Optional[str], port: Optional[int]):
    """Query /v1/models catalog or list from local configuration."""
    console = Console() if HAS_RICH else None
    cfg = get_config()
    target_host = host or cfg.server.host
    target_port = port or cfg.server.port
    models_url = f"http://{target_host}:{target_port}/v1/models"

    remote_models = None
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(models_url)
            if resp.status_code == 200:
                remote_models = resp.json().get("data", [])
    except Exception:
        pass

    if console:
        table = Table(title="Frontier Model Aliases & Failover Target Chains", border_style="cyan")
        table.add_column("Frontier Alias", style="bold green")
        table.add_column("Primary Route", style="cyan")
        table.add_column("Secondary Fallback", style="yellow")
        table.add_column("Tertiary Fallback", style="magenta")

        for alias, targets in cfg.alias_mappings.items():
            t1 = f"{targets[0].provider}/{targets[0].model}" if len(targets) > 0 else "-"
            t2 = f"{targets[1].provider}/{targets[1].model}" if len(targets) > 1 else "-"
            t3 = f"{targets[2].provider}/{targets[2].model}" if len(targets) > 2 else "-"
            table.add_row(alias, t1, t2, t3)

        console.print("\n")
        console.print(table)
        if remote_models:
            console.print(f"\n[dim]Gateway reported {len(remote_models)} live model routes registered.[/dim]\n")
    else:
        for alias, targets in cfg.alias_mappings.items():
            chain = " -> ".join(f"{t.provider}/{t.model}" for t in targets)
            print(f"{alias}: {chain}")


@cli.command("test", help="Execute the automated test suite.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Run pytest in verbose mode")
@click.option("--pattern", "-k", default=None, help="Filter tests by keyword expression")
def run_tests(verbose: bool, pattern: Optional[str]):
    """Invoke pytest subprocess and report test outcomes."""
    args = [sys.executable, "-m", "pytest"]
    if verbose:
        args.append("-v")
    if pattern:
        args.extend(["-k", pattern])

    print(f"[*] Executing test suite: {' '.join(args)}")
    result = subprocess.run(args)
    sys.exit(result.returncode)


@cli.command("compress", help="Test context compression and token multiplier on a prompt.")
@click.option("--file", "-f", "input_file", type=click.Path(exists=True), help="Path to JSON file of chat messages")
@click.option("--turns", "-t", default=4, type=int, help="Preserve recent turns count")
def compress_prompt(input_file: Optional[str], turns: int):
    """Run ContextCompressor on sample messages and display token savings."""
    console = Console() if HAS_RICH else None
    from swarm.compression import CompressionEngine

    engine = CompressionEngine(preserve_recent=turns)

    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
            messages = raw if isinstance(raw, list) else raw.get("messages", [])
    else:
        # Default sample conversation for demo
        messages = [
            {"role": "system", "content": "You are a code optimization assistant with strict safety rules."},
            {"role": "user", "content": "Hello, could you help me write a quick script?"},
            {"role": "assistant", "content": "Certainly! I'd be glad to help. What language and task?"},
            {"role": "user", "content": "Python, parse JSON log file and extract error lines."},
            {"role": "assistant", "content": "Understood! Here is the code:\n```python\nimport json\n```"},
            {"role": "user", "content": "Can you also make it filter by HTTP 429?"},
            {"role": "assistant", "content": "Sure, updated with status_code == 429 check."},
            {"role": "user", "content": "Great, thanks! Run it on test.log."},
        ]

    compressed_msgs, stats = engine.compress(messages)

    savings_pct = (stats.tokens_saved / max(stats.original_tokens, 1)) * 100.0

    if console:
        panel_content = (
            f"Original Messages: {len(messages)} turns\n"
            f"Compressed Messages: {len(compressed_msgs)} turns\n"
            f"Original Tokens: {stats.original_tokens}\n"
            f"Compressed Tokens: {stats.compressed_tokens}\n"
            f"Tokens Saved: {stats.tokens_saved} ({savings_pct:.1f}%)\n"
            f"Token Multiplier: {stats.token_multiplier:.2f}x\n"
            f"Prefix Cache Hit: {stats.prefix_cache_hit}\n"
            f"Latency: {stats.duration_ms:.2f} ms"
        )
        console.print(Panel(panel_content, title="Context Compression Metrics", border_style="green"))
    else:
        print(f"Original turns: {len(messages)} -> Compressed: {len(compressed_msgs)}")
        print(f"Original tokens: {stats.original_tokens} -> Compressed: {stats.compressed_tokens}")
        print(f"Token Multiplier: {stats.token_multiplier:.2f}x (Saved: {stats.tokens_saved})")



def main() -> None:
    """Entry point for swarm executable."""
    cli()


if __name__ == "__main__":
    main()
