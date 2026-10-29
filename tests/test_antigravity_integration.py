"""Automated Verification Suite for Swarm Agent Antigravity Integration.

Tests:
1. MCP Protocol Server (JSON-RPC 2.0 stdio handling, tool catalog, method dispatch)
2. Tool execution across all 6 Swarm MCP tools
3. Attachment and detachment lifecycle in an isolated sandbox
4. Clean restoration and zero-residue verification
5. Real subprocess stdio communication
"""

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pytest

from swarm.antigravity import AntigravityIntegrationManager
from swarm.config import get_config
from swarm.mcp_server import MCP_TOOLS, PROTOCOL_VERSION, SERVER_NAME, SwarmMCPServer


@pytest.fixture
def mcp_server():
    """Create an instance of SwarmMCPServer."""
    return SwarmMCPServer()


@pytest.mark.asyncio
async def test_mcp_initialize(mcp_server):
    """Verify MCP initialize returns standard protocol version and server info."""
    req = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "antigravity", "version": "1.0"},
        },
    }
    resp = await mcp_server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 100
    assert "result" in resp
    assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert resp["result"]["serverInfo"]["name"] == SERVER_NAME
    assert "tools" in resp["result"]["capabilities"]


@pytest.mark.asyncio
async def test_mcp_ping(mcp_server):
    """Verify ping returns empty result."""
    req = {"jsonrpc": "2.0", "id": 101, "method": "ping", "params": {}}
    resp = await mcp_server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 101
    assert resp["result"] == {}


@pytest.mark.asyncio
async def test_mcp_notifications(mcp_server):
    """Verify notifications/initialized requires no reply."""
    req = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }
    resp = await mcp_server.handle_request(req)
    assert resp is None


@pytest.mark.asyncio
async def test_mcp_tools_list(mcp_server):
    """Verify all 6 Swarm tools are exposed with valid schemas."""
    req = {"jsonrpc": "2.0", "id": 102, "method": "tools/list", "params": {}}
    resp = await mcp_server.handle_request(req)
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}

    expected_tools = {
        "swarm_chat",
        "swarm_run_autonomous_loop",
        "swarm_model_catalog",
        "swarm_system_status",
        "swarm_powershell_exec",
        "swarm_compress_context",
    }
    assert expected_tools.issubset(tool_names)
    for t in tools:
        assert "description" in t
        assert "inputSchema" in t


@pytest.mark.asyncio
async def test_mcp_unknown_method(mcp_server):
    """Verify unknown method returns error code -32601."""
    req = {"jsonrpc": "2.0", "id": 103, "method": "unknown_action", "params": {}}
    resp = await mcp_server.handle_request(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_mcp_tool_system_status(mcp_server):
    """Verify swarm_system_status returns telemetry and provider matrix."""
    req = {
        "jsonrpc": "2.0",
        "id": 104,
        "method": "tools/call",
        "params": {"name": "swarm_system_status", "arguments": {}},
    }
    resp = await mcp_server.handle_request(req)
    assert not resp["result"]["isError"]
    text = resp["result"]["content"][0]["text"]
    assert "=== Swarm Agent Live System Status ===" in text
    assert "Provider Matrix:" in text


@pytest.mark.asyncio
async def test_mcp_tool_model_catalog(mcp_server):
    """Verify swarm_model_catalog returns 7 frontier model aliases."""
    req = {
        "jsonrpc": "2.0",
        "id": 105,
        "method": "tools/call",
        "params": {"name": "swarm_model_catalog", "arguments": {}},
    }
    resp = await mcp_server.handle_request(req)
    assert not resp["result"]["isError"]
    text = resp["result"]["content"][0]["text"]
    assert "deepseek-v4-pro" in text
    assert "kimi-k3" in text
    assert "glm-5.3" in text
    assert "qwen-3.8" in text
    assert "gemini-3.8-flash-high" in text


@pytest.mark.asyncio
async def test_mcp_tool_compress_context(mcp_server):
    """Verify context compression tool calculates savings."""
    req = {
        "jsonrpc": "2.0",
        "id": 106,
        "method": "tools/call",
        "params": {
            "name": "swarm_compress_context",
            "arguments": {
                "messages": [
                    {"role": "system", "content": "You are a code assistant."},
                    {"role": "user", "content": "Hello there! Can you please help me write a function?"},
                    {"role": "assistant", "content": "Certainly! I'd be delighted to help you write that function."},
                    {"role": "user", "content": "Write def add(a, b): return a + b."},
                ]
            },
        },
    }
    resp = await mcp_server.handle_request(req)
    assert not resp["result"]["isError"]
    text = resp["result"]["content"][0]["text"]
    assert "Token Multiplier" in text
    assert "Compressed Tokens" in text


@pytest.mark.asyncio
async def test_mcp_tool_powershell_exec(mcp_server):
    """Verify guarded PowerShell bridge execution through MCP."""
    req = {
        "jsonrpc": "2.0",
        "id": 107,
        "method": "tools/call",
        "params": {
            "name": "swarm_powershell_exec",
            "arguments": {
                "command": "Write-Output 'Antigravity-Swarm-Bridge-OK'",
            },
        },
    }
    resp = await mcp_server.handle_request(req)
    assert not resp["result"]["isError"]
    text = resp["result"]["content"][0]["text"]
    assert "Exit Code: 0" in text
    assert "Antigravity-Swarm-Bridge-OK" in text


@pytest.mark.asyncio
async def test_mcp_tool_run_autonomous_loop(mcp_server):
    """Verify 3-tier feedback loop execution report."""
    req = {
        "jsonrpc": "2.0",
        "id": 108,
        "method": "tools/call",
        "params": {
            "name": "swarm_run_autonomous_loop",
            "arguments": {
                "task_description": "Implement and verify fast sorting algorithm",
                "max_tier1_budget": 3,
            },
        },
    }
    resp = await mcp_server.handle_request(req)
    assert not resp["result"]["isError"]
    text = resp["result"]["content"][0]["text"]
    assert "Tier 1: Local Test Loop" in text
    assert "Tier 2: Middle Adversarial Audit Loop" in text
    assert "Tier 3: Outer Goal Convergence Loop" in text


def test_antigravity_attach_and_detach_sandbox():
    """Verify attach, status, detach, and zero residue in an isolated mock sandbox."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mock_config_dir = tmp_path / ".gemini" / "config"
        mock_config_dir.mkdir(parents=True)

        mock_mcp_file = mock_config_dir / "mcp_config.json"
        initial_mcp_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                },
                "memory": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-memory"],
                },
            }
        }
        with open(mock_mcp_file, "w", encoding="utf-8") as f:
            json.dump(initial_mcp_data, f, indent=2)

        mock_app_config = mock_config_dir / "config.json"
        with open(mock_app_config, "w", encoding="utf-8") as f:
            json.dump({"plugins": {"existing-plugin": {}}}, f, indent=2)

        manager = AntigravityIntegrationManager(config_dir=mock_config_dir)

        # 1. Verify initially not attached
        assert not manager.is_attached()
        status_before = manager.get_status()
        assert not status_before["attached"]

        # 2. Attach
        attach_res = manager.attach()
        assert attach_res["status"] == "ATTACHED"
        assert manager.is_attached()

        # Check mcp_config.json contains swarm-agent AND original servers
        with open(mock_mcp_file, "r", encoding="utf-8") as f:
            updated_mcp = json.load(f)
        assert "swarm-agent" in updated_mcp["mcpServers"]
        assert "filesystem" in updated_mcp["mcpServers"]
        assert "memory" in updated_mcp["mcpServers"]

        # Check plugin directory deployed
        installed_plugin = mock_config_dir / "plugins" / "swarm-agent-plugin"
        assert installed_plugin.is_dir()
        assert (installed_plugin / "plugin.json").is_file()
        assert (installed_plugin / "skills" / "swarm-agent" / "SKILL.md").is_file()

        # Check config.json updated
        with open(mock_app_config, "r", encoding="utf-8") as f:
            updated_app_cfg = json.load(f)
        assert "swarm-agent-plugin" in updated_app_cfg["plugins"]
        assert "existing-plugin" in updated_app_cfg["plugins"]

        # 3. Detach
        detach_res = manager.detach()
        assert detach_res["status"] == "DETACHED"
        assert detach_res["clean"] is True
        assert not manager.is_attached()

        # Verify mcp_config.json has swarm-agent removed but original servers intact
        with open(mock_mcp_file, "r", encoding="utf-8") as f:
            final_mcp = json.load(f)
        assert "swarm-agent" not in final_mcp["mcpServers"]
        assert "filesystem" in final_mcp["mcpServers"]
        assert "memory" in final_mcp["mcpServers"]

        # Verify plugin directory completely deleted
        assert not installed_plugin.exists()

        # Verify config.json has plugin removed
        with open(mock_app_config, "r", encoding="utf-8") as f:
            final_app_cfg = json.load(f)
        assert "swarm-agent-plugin" not in final_app_cfg["plugins"]
        assert "existing-plugin" in final_app_cfg["plugins"]

        # Verify clean check
        is_clean, issues = manager.verify_clean()
        assert is_clean, f"Expected clean environment, got issues: {issues}"


def test_attach_idempotency_sandbox():
    """Verify calling attach multiple times doesn't corrupt configs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mock_config_dir = tmp_path / ".gemini" / "config"
        mock_config_dir.mkdir(parents=True)
        mock_mcp_file = mock_config_dir / "mcp_config.json"
        with open(mock_mcp_file, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": {}}, f)

        manager = AntigravityIntegrationManager(config_dir=mock_config_dir)

        manager.attach()
        manager.attach()

        with open(mock_mcp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "swarm-agent" in data["mcpServers"]
        assert len(data["mcpServers"]) == 1

        manager.detach()
        manager.detach()
        is_clean, _ = manager.verify_clean()
        assert is_clean


def test_mcp_stdio_subprocess_e2e():
    """Verify real MCP server stdio process initializes and responds to JSON-RPC messages."""
    python_exe = sys.executable

    proc = subprocess.Popen(
        [python_exe, "-m", "swarm.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    try:
        # Send initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()

        raw_line = proc.stdout.readline()
        assert raw_line, "Expected JSON-RPC response from server"
        resp = json.loads(raw_line)
        assert resp["id"] == 1
        assert resp["result"]["serverInfo"]["name"] == "swarm-agent"

        # Send tools/list
        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        proc.stdin.write(json.dumps(tools_req) + "\n")
        proc.stdin.flush()

        raw_tools_line = proc.stdout.readline()
        tools_resp = json.loads(raw_tools_line)
        assert tools_resp["id"] == 2
        tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
        assert "swarm_chat" in tool_names
        assert "swarm_system_status" in tool_names

    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.asyncio
async def test_mcp_tool_run_autonomous_loop_security_rejection(mcp_server):
    """Verify 3-tier loop Tier 2 Auditor rejects insecure code and generates Reasoner plan."""
    insecure_code = (
        "import os\n"
        "def dangerous(cmd):\n"
        "    return eval(cmd)\n"
    )
    req = {
        "jsonrpc": "2.0",
        "id": 109,
        "method": "tools/call",
        "params": {
            "name": "swarm_run_autonomous_loop",
            "arguments": {
                "task_description": "Validate security handling of eval expressions",
                "code_context": insecure_code,
            },
        },
    }
    resp = await mcp_server.handle_request(req)
    assert not resp["result"]["isError"]
    text = resp["result"]["content"][0]["text"]
    assert "Verdict: REJECT" in text
    assert "SEC-03-DYNAMIC-EVAL" in text
    assert "Reasoner Remediation Plan:" in text


def test_antigravity_byte_for_byte_backup_restoration():
    """Verify that attach + detach restores configs byte-for-byte with matching hashes."""
    import hashlib

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mock_config_dir = tmp_path / ".gemini" / "config"
        mock_config_dir.mkdir(parents=True)

        mock_mcp_file = mock_config_dir / "mcp_config.json"
        original_mcp_content = '{\n    "mcpServers":   {\n        "custom_server": {"command": "cmd"}\n    }\n}\n'
        mock_mcp_file.write_text(original_mcp_content, encoding="utf-8")
        orig_mcp_hash = hashlib.sha256(mock_mcp_file.read_bytes()).hexdigest()

        mock_cfg_file = mock_config_dir / "config.json"
        original_cfg_content = '{\n    "plugins":   {\n        "custom_plugin": {}\n    }\n}\n'
        mock_cfg_file.write_text(original_cfg_content, encoding="utf-8")
        orig_cfg_hash = hashlib.sha256(mock_cfg_file.read_bytes()).hexdigest()

        manager = AntigravityIntegrationManager(config_dir=mock_config_dir)
        manager.attach()
        assert manager.is_attached()

        manager.detach()
        assert not manager.is_attached()

        # Verify byte-for-byte restoration
        restored_mcp_hash = hashlib.sha256(mock_mcp_file.read_bytes()).hexdigest()
        assert restored_mcp_hash == orig_mcp_hash
        assert mock_mcp_file.read_text(encoding="utf-8") == original_mcp_content

        restored_cfg_hash = hashlib.sha256(mock_cfg_file.read_bytes()).hexdigest()
        assert restored_cfg_hash == orig_cfg_hash
        assert mock_cfg_file.read_text(encoding="utf-8") == original_cfg_content

        clean, issues = manager.verify_clean()
        assert clean, f"Expected clean environment, got: {issues}"


def test_antigravity_clean_when_no_prior_configs():
    """Verify that if configs did not exist prior to attach, detach leaves zero files behind."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mock_config_dir = tmp_path / ".gemini" / "config"

        manager = AntigravityIntegrationManager(config_dir=mock_config_dir)
        manager.attach()
        assert manager.is_attached()

        manager.detach()
        assert not manager.is_attached()

        # Both mcp_config.json and config.json should NOT exist
        assert not (mock_config_dir / "mcp_config.json").exists()
        assert not (mock_config_dir / "config.json").exists()
        assert not (mock_config_dir / "plugins" / "swarm-agent-plugin").exists()

        clean, issues = manager.verify_clean()
        assert clean, f"Expected clean, got issues: {issues}"

