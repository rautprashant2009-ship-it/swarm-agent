"""Antigravity Integration Manager for Swarm Agent.

Manages temporary attachment and zero-residue detachment of Swarm Agent
into the local Antigravity environment (MCP servers and skills/plugins).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("swarm.antigravity")

# Default Antigravity configuration paths
DEFAULT_GEMINI_DIR = Path(os.environ.get("USERPROFILE", "C:\\Users\\Admin")) / ".gemini"
CONFIG_DIR = DEFAULT_GEMINI_DIR / "config"
MCP_CONFIG_PATH = CONFIG_DIR / "mcp_config.json"
ANTIGRAVITY_CONFIG_PATH = CONFIG_DIR / "config.json"
PLUGINS_DIR = CONFIG_DIR / "plugins"
TARGET_PLUGIN_DIR = PLUGINS_DIR / "swarm-agent-plugin"

# Swarm paths
SWARM_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_SOURCE_DIR = SWARM_ROOT / "antigravity_plugin"
STATE_FILE = SWARM_ROOT / ".antigravity_attached.json"

# Backups
MCP_BACKUP_PATH = CONFIG_DIR / "mcp_config.json.bak_swarm"
CONFIG_BACKUP_PATH = CONFIG_DIR / "config.json.bak_swarm"


def get_venv_python() -> Path:
    """Locate the Python executable in the local virtual environment."""
    venv_py = SWARM_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        return venv_py
    # Fallback to current sys.executable
    return Path(sys.executable)


def compute_file_hash(path: Path) -> str:
    """Calculate SHA-256 of a file."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class AntigravityIntegrationManager:
    """Coordinates seamless temporary attach, verification, and detachment."""

    def __init__(
        self,
        swarm_root: Optional[Path] = None,
        config_dir: Optional[Path] = None,
    ):
        self.swarm_root = swarm_root or SWARM_ROOT
        self.config_dir = config_dir or CONFIG_DIR
        self.mcp_config_path = self.config_dir / "mcp_config.json"
        self.antigravity_config_path = self.config_dir / "config.json"
        self.plugins_dir = self.config_dir / "plugins"
        self.target_plugin_dir = self.plugins_dir / "swarm-agent-plugin"
        self.plugin_source_dir = self.swarm_root / "antigravity_plugin"
        self.state_file = self.swarm_root / ".antigravity_attached.json"

        self.mcp_backup_path = self.config_dir / "mcp_config.json.bak_swarm"
        self.config_backup_path = self.config_dir / "config.json.bak_swarm"

    def is_attached(self) -> bool:
        """Check whether Swarm is currently registered in Antigravity."""
        mcp_attached = False
        if self.mcp_config_path.is_file():
            try:
                with open(self.mcp_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    mcp_attached = "swarm-agent" in data.get("mcpServers", {})
            except Exception:
                pass

        plugin_attached = self.target_plugin_dir.is_dir()
        return mcp_attached or plugin_attached

    def get_status(self) -> Dict[str, Any]:
        """Inspect current attachment and health status."""
        attached = self.is_attached()
        mcp_present = False
        if self.mcp_config_path.is_file():
            try:
                with open(self.mcp_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    mcp_present = "swarm-agent" in data.get("mcpServers", {})
            except Exception:
                pass

        plugin_present = self.target_plugin_dir.is_dir()
        config_present = False
        if self.antigravity_config_path.is_file():
            try:
                with open(self.antigravity_config_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    config_present = "swarm-agent-plugin" in cdata.get("plugins", {})
            except Exception:
                pass

        state_data = {}
        if self.state_file.is_file():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
            except Exception:
                pass

        return {
            "attached": attached,
            "mcp_server_registered": mcp_present,
            "plugin_installed": plugin_present,
            "config_registered": config_present,
            "mcp_config_path": str(self.mcp_config_path),
            "plugin_path": str(self.target_plugin_dir),
            "state_metadata": state_data,
        }

    def attach(self) -> Dict[str, Any]:
        """Temporarily attach Swarm Agent into Antigravity."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        venv_py = get_venv_python()

        # 1. Backup and update mcp_config.json
        mcp_existed = self.mcp_config_path.is_file()
        if mcp_existed:
            if not self.mcp_backup_path.is_file():
                shutil.copy2(self.mcp_config_path, self.mcp_backup_path)
            with open(self.mcp_config_path, "r", encoding="utf-8") as f:
                mcp_data = json.load(f)
        else:
            mcp_data = {"mcpServers": {}}

        if "mcpServers" not in mcp_data:
            mcp_data["mcpServers"] = {}

        mcp_data["mcpServers"]["swarm-agent"] = {
            "command": str(venv_py),
            "args": [
                "-m",
                "swarm.mcp_server",
            ],
            "cwd": str(self.swarm_root),
        }

        with open(self.mcp_config_path, "w", encoding="utf-8") as f:
            json.dump(mcp_data, f, indent=2)

        # 2. Deploy plugin folder
        if self.plugin_source_dir.is_dir():
            if self.target_plugin_dir.exists():
                shutil.rmtree(self.target_plugin_dir)
            shutil.copytree(self.plugin_source_dir, self.target_plugin_dir)

        # 3. Register plugin in config.json (with backup)
        config_existed = self.antigravity_config_path.is_file()
        if config_existed:
            if not self.config_backup_path.is_file():
                shutil.copy2(self.antigravity_config_path, self.config_backup_path)
            try:
                with open(self.antigravity_config_path, "r", encoding="utf-8") as f:
                    antigravity_cfg = json.load(f)
                if "plugins" not in antigravity_cfg:
                    antigravity_cfg["plugins"] = {}
                antigravity_cfg["plugins"]["swarm-agent-plugin"] = {
                    "installedFrom": {
                        "id": "local/swarm-agent-plugin",
                        "marketplace": "local",
                        "version": "1.0.0",
                    }
                }
                with open(self.antigravity_config_path, "w", encoding="utf-8") as f:
                    json.dump(antigravity_cfg, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not update config.json: {e}")

        # 4. Save state file
        state = {
            "attached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "swarm_root": str(self.swarm_root),
            "python_executable": str(venv_py),
            "mcp_config_existed": mcp_existed,
            "mcp_config_backup": str(self.mcp_backup_path) if mcp_existed else None,
            "config_existed": config_existed,
            "config_backup": str(self.config_backup_path) if config_existed else None,
            "plugin_installed_path": str(self.target_plugin_dir),
            "active": True,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        return {
            "status": "ATTACHED",
            "mcp_server": "swarm-agent",
            "plugin": "swarm-agent-plugin",
            "python": str(venv_py),
            "state_file": str(self.state_file),
        }

    def detach(self) -> Dict[str, Any]:
        """Completely and cleanly detach Swarm Agent from Antigravity."""
        changes_reverted = []
        state_data = {}
        if self.state_file.is_file():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
            except Exception:
                pass

        mcp_existed = state_data.get("mcp_config_existed", True)
        config_existed = state_data.get("config_existed", True)

        # 1. Restore or clean mcp_config.json
        if self.mcp_backup_path.is_file():
            try:
                shutil.copy2(self.mcp_backup_path, self.mcp_config_path)
                self.mcp_backup_path.unlink(missing_ok=True)
                changes_reverted.append("Restored mcp_config.json byte-for-byte from pre-attach backup")
            except Exception as e:
                logger.warning(f"Error restoring mcp_config.json from backup: {e}")
        elif self.mcp_config_path.is_file():
            if not mcp_existed:
                try:
                    self.mcp_config_path.unlink(missing_ok=True)
                    changes_reverted.append("Removed created mcp_config.json (did not exist before attach)")
                except Exception as e:
                    logger.warning(f"Error removing created mcp_config.json: {e}")
            else:
                try:
                    with open(self.mcp_config_path, "r", encoding="utf-8") as f:
                        mcp_data = json.load(f)
                    if "swarm-agent" in mcp_data.get("mcpServers", {}):
                        del mcp_data["mcpServers"]["swarm-agent"]
                        with open(self.mcp_config_path, "w", encoding="utf-8") as f:
                            json.dump(mcp_data, f, indent=2)
                        changes_reverted.append("Removed 'swarm-agent' from mcp_config.json")
                except Exception as e:
                    logger.warning(f"Error updating mcp_config.json: {e}")

        # 2. Restore or clean config.json
        if self.config_backup_path.is_file():
            try:
                shutil.copy2(self.config_backup_path, self.antigravity_config_path)
                self.config_backup_path.unlink(missing_ok=True)
                changes_reverted.append("Restored config.json byte-for-byte from pre-attach backup")
            except Exception as e:
                logger.warning(f"Error restoring config.json from backup: {e}")
        elif self.antigravity_config_path.is_file():
            if not config_existed:
                try:
                    self.antigravity_config_path.unlink(missing_ok=True)
                    changes_reverted.append("Removed created config.json (did not exist before attach)")
                except Exception as e:
                    logger.warning(f"Error removing created config.json: {e}")
            else:
                try:
                    with open(self.antigravity_config_path, "r", encoding="utf-8") as f:
                        antigravity_cfg = json.load(f)
                    if "swarm-agent-plugin" in antigravity_cfg.get("plugins", {}):
                        del antigravity_cfg["plugins"]["swarm-agent-plugin"]
                        with open(self.antigravity_config_path, "w", encoding="utf-8") as f:
                            json.dump(antigravity_cfg, f, indent=2)
                        changes_reverted.append("Removed 'swarm-agent-plugin' from config.json")
                except Exception as e:
                    logger.warning(f"Error updating config.json: {e}")

        # 3. Remove installed plugin directory
        if self.target_plugin_dir.exists():
            shutil.rmtree(self.target_plugin_dir, ignore_errors=True)
            changes_reverted.append(f"Removed {self.target_plugin_dir}")

        # 4. Remove state file
        if self.state_file.is_file():
            self.state_file.unlink(missing_ok=True)
            changes_reverted.append("Removed .antigravity_attached.json")

        return {
            "status": "DETACHED",
            "clean": True,
            "reverted_actions": changes_reverted,
        }

    def verify_clean(self) -> Tuple[bool, List[str]]:
        """Verify that zero swarm residue remains in Antigravity."""
        issues = []
        if self.mcp_config_path.is_file():
            try:
                with open(self.mcp_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "swarm-agent" in data.get("mcpServers", {}):
                        issues.append("'swarm-agent' still present in mcp_config.json")
            except Exception as e:
                issues.append(f"Could not read mcp_config.json: {e}")

        if self.target_plugin_dir.exists():
            issues.append(f"Plugin directory still exists: {self.target_plugin_dir}")

        if self.antigravity_config_path.is_file():
            try:
                with open(self.antigravity_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "swarm-agent-plugin" in data.get("plugins", {}):
                        issues.append("'swarm-agent-plugin' still present in config.json")
            except Exception:
                pass

        if self.mcp_backup_path.is_file():
            issues.append(f"MCP backup file still exists: {self.mcp_backup_path}")

        if self.config_backup_path.is_file():
            issues.append(f"Config backup file still exists: {self.config_backup_path}")

        if self.state_file.is_file():
            issues.append(f"State file still exists: {self.state_file}")

        return (len(issues) == 0, issues)


def run_antigravity_cli():
    """CLI handler for `python -m swarm antigravity`."""
    import sys

    manager = AntigravityIntegrationManager()
    args = sys.argv[1:]
    cmd = args[1] if len(args) > 1 and args[0] == "antigravity" else (args[0] if args else "status")

    if cmd == "attach":
        res = manager.attach()
        print("==================================================================")
        print("   SWARM AGENT: Attached to Antigravity (TEMPORARY)")
        print("==================================================================")
        print(f"[*] MCP Server:     {res['mcp_server']} (registered in mcp_config.json)")
        print(f"[*] Skill / Plugin: {res['plugin']}")
        print(f"[*] Python Runtime: {res['python']}")
        print(f"[*] State Tracker:  {res['state_file']}")
        print("\n[+] To detach cleanly at any time, run: detach_from_antigravity.bat")
        print("==================================================================")

    elif cmd == "detach":
        res = manager.detach()
        clean, issues = manager.verify_clean()
        print("==================================================================")
        print("   SWARM AGENT: Detached from Antigravity")
        print("==================================================================")
        for act in res["reverted_actions"]:
            print(f"[-] {act}")
        if clean:
            print("\n[SUCCESS] Antigravity environment completely restored to original state!")
            print("[SUCCESS] Zero residue left behind.")
        else:
            print(f"\n[WARNING] Remaining issues: {issues}")
        print("==================================================================")

    elif cmd == "status":
        status = manager.get_status()
        print("==================================================================")
        print("   SWARM AGENT: Antigravity Integration Status")
        print("==================================================================")
        print(f"[*] Attached:              {status['attached']}")
        print(f"[*] MCP Server Registered: {status['mcp_server_registered']}")
        print(f"[*] Plugin Installed:      {status['plugin_installed']}")
        print(f"[*] Config Registered:      {status['config_registered']}")
        print("==================================================================")

    elif cmd == "test":
        print("[*] Testing MCP Server via stdio subprocess...")
        from swarm.mcp_server import SwarmMCPServer
        import asyncio

        server = SwarmMCPServer()
        init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = asyncio.run(server.handle_request(init_req))
        assert resp and resp["result"]["serverInfo"]["name"] == "swarm-agent"
        print("[+] Initialize passed!")

        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        tools_resp = asyncio.run(server.handle_request(tools_req))
        tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
        print(f"[+] Tools registered ({len(tool_names)}): {', '.join(tool_names)}")

        status_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "swarm_system_status", "arguments": {}},
        }
        status_resp = asyncio.run(server.handle_request(status_req))
        print("[+] System status tool output:")
        print(status_resp["result"]["content"][0]["text"])
        print("\n[SUCCESS] Antigravity MCP Integration Test passed!")

    else:
        print(f"Unknown command: {cmd}. Usage: python -m swarm antigravity [attach|detach|status|test]")


if __name__ == "__main__":
    run_antigravity_cli()
