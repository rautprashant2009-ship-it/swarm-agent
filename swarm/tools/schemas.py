"""OpenAI-Compatible Tool Calling Schemas & Dispatcher for Swarm Agent.

Exports standard OpenAI function calling tool definitions for the swarm runtime:
- `run_command`
- `read_file`
- `write_file`
- `edit_file`
- `list_directory`
- `search_files`
- `run_python_tests`
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Dict, List, Optional

from swarm.tools.executor import run_command
from swarm.tools.files import edit_file, list_directory, read_file, search_files, write_file
from swarm.tools.runner import run_python_tests

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command via PowerShell on the local Windows host machine with timeout protection and streaming output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact PowerShell command line string to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory path. Defaults to workspace root if omitted.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds before process termination. Default: 60.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text content from a file with line-based pagination and binary file protection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based starting line number. Default: 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Default: 500.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite content to a file atomically, automatically creating parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the destination file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete text content to write.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite existing file. Default: true.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Perform precise single-block replacement in an existing file with uniqueness validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the target file.",
                    },
                    "target_content": {
                        "type": "string",
                        "description": "The exact contiguous block of characters to be replaced.",
                    },
                    "replacement_content": {
                        "type": "string",
                        "description": "The new replacement text to insert.",
                    },
                    "allow_multiple": {
                        "type": "boolean",
                        "description": "If true, replaces all occurrences. If false, errors if more than one occurrence is found. Default: false.",
                    },
                },
                "required": ["path", "target_content", "replacement_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories within a directory, returning file sizes, types, and modification times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to traverse subdirectories recursively. Default: false.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum recursive directory depth. Default: 2.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text or regular expressions across files in a directory tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Base directory to search in.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The search string or regular expression.",
                    },
                    "is_regex": {
                        "type": "boolean",
                        "description": "Whether to treat query as a regular expression. Default: false.",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern for filtering filenames (e.g. '*.py'). Default: '*'.",
                    },
                },
                "required": ["directory", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_tests",
            "description": "Run the project test suite using pytest or unittest, returning parsed failure diagnostics and exit codes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_target": {
                        "type": "string",
                        "description": "Specific test file or test directory path. Defaults to 'tests/' if omitted.",
                    },
                    "test_name": {
                        "type": "string",
                        "description": "Optional specific test function or class name filter.",
                    },
                },
            },
        },
    },
]


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return a deep copy of the registered tool schemas."""
    return copy.deepcopy(TOOL_SCHEMAS)


def get_tool_map() -> Dict[str, Callable[..., Any]]:
    """Return a mapping of tool names to callable execution functions."""
    return {
        "run_command": run_command,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "list_directory": list_directory,
        "search_files": search_files,
        "run_python_tests": run_python_tests,
    }


def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Safely dispatch an OpenAI tool call by name with argument validation.

    Args:
        tool_name: Name of registered function.
        arguments: Dict of keyword arguments passed from model.

    Returns:
        Structured result dict matching tool output or standard error envelope.
    """
    tool_map = get_tool_map()
    if tool_name not in tool_map:
        return {
            "error": f"Tool '{tool_name}' not found. Available tools: {list(tool_map.keys())}"
        }

    func = tool_map[tool_name]

    try:
        result = func(**arguments)
        if hasattr(result, "to_dict"):
            return result.to_dict()
        elif hasattr(result, "model_dump"):
            return result.model_dump()
        elif isinstance(result, dict):
            return result
        else:
            return {"result": result}
    except Exception as exc:
        return {"error": f"Execution of tool '{tool_name}' failed: {exc}"}
