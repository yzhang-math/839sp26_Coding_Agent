"""
Pure library of tool implementations and tool definitions.
No state, no LLM logic. Used by main_agent and sub_agent.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (minimal: only what tool impls need)
# ---------------------------------------------------------------------------
CONFIG_FILENAMES = ("config.json",)
_script_dir = Path(__file__).resolve().parent


def _load_config() -> dict:
    """Load config from config.json. Used for SHELL_TIMEOUT in run_shell_command_impl."""
    for base in (_script_dir, Path.cwd()):
        for name in CONFIG_FILENAMES:
            path = base / name
            if path.is_file():
                try:
                    with open(path, encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
    return {}


_config = _load_config()
SHELL_TIMEOUT = _config.get("shell_timeout", 15)


# ---------------------------------------------------------------------------
# Sandbox & file/shell helpers
# ---------------------------------------------------------------------------

def _sandbox_path(sandbox_dir: str, filename: str) -> Path:
    """Normalize filename to sandbox: prepend sandbox path, resolve relative path. Ensures path stays within sandbox."""
    if not filename or filename.strip() == "":
        raise ValueError("Filename cannot be empty")
    sandbox_root = Path(sandbox_dir).resolve()
    path = (sandbox_root / filename).resolve()
    try:
        path.relative_to(sandbox_root)
    except ValueError:
        raise ValueError(f"Path {filename} is outside the sandbox directory")
    return path


def read_file_impl(sandbox_dir: str, filename: str) -> str:
    path = _sandbox_path(sandbox_dir, filename)
    if not path.exists():
        return f"Error: File '{filename}' does not exist."
    if path.is_dir():
        return f"Error: '{filename}' is a directory, not a file. Use a file path instead."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except PermissionError:
        return f"Error: Permission denied reading '{filename}'."
    except Exception as e:
        return f"Error reading '{filename}': {str(e)}"


def write_file_impl(sandbox_dir: str, filename: str, content: str) -> str:
    if not filename or filename.strip() == "":
        return "Error: Filename cannot be empty."
    try:
        path = _sandbox_path(sandbox_dir, filename)
        if path.exists() and path.is_dir():
            return f"Error: '{filename}' is a directory. Cannot write to a directory."
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {filename}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except PermissionError:
        return f"Error: Permission denied writing to '{filename}'."
    except Exception as e:
        return f"Error writing to '{filename}': {str(e)}"


def run_shell_command_impl(sandbox_dir: str, command: str) -> str:
    result = subprocess.run(
        command,
        shell=True,
        cwd=sandbox_dir,
        timeout=SHELL_TIMEOUT,
        capture_output=True,
        text=True,
    )
    out = result.stdout or ""
    err = result.stderr or ""
    if result.returncode != 0:
        return f"Exit code: {result.returncode}\nstdout:\n{out}\nstderr:\n{err}"
    return out or "(no output)"


def move_file_impl(sandbox_dir: str, deliverables_dir: str, source: str, dest: str) -> str:
    """Move a file from sandbox to deliverables folder."""
    source_path = _sandbox_path(sandbox_dir, source)
    if not source_path.exists():
        return f"Error: Source file '{source}' does not exist."
    if source_path.is_dir():
        return f"Error: '{source}' is a directory, not a file."
    dest_path = Path(deliverables_dir) / dest
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(dest_path))
    return f"Moved '{source}' to deliverables/{dest}"


def copy_file_impl(sandbox_dir: str, deliverables_dir: str, source: str, dest: str) -> str:
    """Copy a file from sandbox to deliverables folder."""
    source_path = _sandbox_path(sandbox_dir, source)
    if not source_path.exists():
        return f"Error: Source file '{source}' does not exist."
    if source_path.is_dir():
        return f"Error: '{source}' is a directory, not a file."
    dest_path = Path(deliverables_dir) / dest
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(dest_path))
    return f"Copied '{source}' to deliverables/{dest}"


def delete_file_impl(sandbox_dir: str, filename: str) -> str:
    """Delete a file from the sandbox."""
    path = _sandbox_path(sandbox_dir, filename)
    if not path.exists():
        return f"Error: File '{filename}' does not exist."
    if path.is_dir():
        return f"Error: '{filename}' is a directory. Use delete_directory tool instead."
    path.unlink()
    return f"Deleted '{filename}'"


# ---------------------------------------------------------------------------
# Sub-Agent tool definitions and executor
# ---------------------------------------------------------------------------

SUB_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the sandbox. Use a path relative to the sandbox (e.g. 'script.py' or 'src/main.py').",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "Path to the file relative to the sandbox."}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the sandbox. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Path to the file relative to the sandbox."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a shell command in the sandbox (e.g. pytest, python script.py). Timeout 15s.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to run in the sandbox."}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move a file from sandbox to the deliverables folder (for handing over to Main Agent).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source file path relative to sandbox."},
                    "dest": {"type": "string", "description": "Destination file path relative to deliverables folder."},
                },
                "required": ["source", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copy a file from sandbox to the deliverables folder (for handing over to Main Agent).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source file path relative to sandbox."},
                    "dest": {"type": "string", "description": "Destination file path relative to deliverables folder."},
                },
                "required": ["source", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file from the sandbox. Use this to clean up temporary files.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "File path relative to sandbox."}},
                "required": ["filename"],
            },
        },
    },
]


def execute_sub_agent_tool(sandbox_dir: str, deliverables_dir: str, name: str, arguments: dict) -> str:
    try:
        if name == "read_file":
            filename = arguments.get("filename", "")
            if not filename:
                return "Error: 'filename' argument is required for read_file."
            return read_file_impl(sandbox_dir, filename)
        if name == "write_file":
            filename = arguments.get("filename", "")
            content = arguments.get("content", "")
            if not filename:
                return "Error: 'filename' argument is required for write_file."
            return write_file_impl(sandbox_dir, filename, content)
        if name == "run_shell_command":
            command = arguments.get("command", "")
            if not command:
                return "Error: 'command' argument is required for run_shell_command."
            return run_shell_command_impl(sandbox_dir, command)
        if name == "move_file":
            source = arguments.get("source", "")
            dest = arguments.get("dest", "")
            if not source or not dest:
                return "Error: 'source' and 'dest' arguments are required for move_file."
            return move_file_impl(sandbox_dir, deliverables_dir, source, dest)
        if name == "copy_file":
            source = arguments.get("source", "")
            dest = arguments.get("dest", "")
            if not source or not dest:
                return "Error: 'source' and 'dest' arguments are required for copy_file."
            return copy_file_impl(sandbox_dir, deliverables_dir, source, dest)
        if name == "delete_file":
            filename = arguments.get("filename", "")
            if not filename:
                return "Error: 'filename' argument is required for delete_file."
            return delete_file_impl(sandbox_dir, filename)
        return f"Unknown tool: {name}"
    except KeyError as e:
        return f"Error: Missing required argument: {str(e)}"
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


# ---------------------------------------------------------------------------
# Main Agent helpers: session files, deliverables, memo
# ---------------------------------------------------------------------------

def write_project_file_impl(session_root: str, filepath: str, content: str) -> str:
    """Write content to a file in the session folder (session root)."""
    if not session_root:
        return "Error: No session workspace."
    root = Path(session_root).resolve()
    path = (root / filepath).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return "Error: filepath must be inside the session folder."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {filepath} in session folder"


def move_file_main_agent_impl(session_root: str, source: str, dest: str) -> str:
    """Move a file within the session folder (paths relative to session root)."""
    if not session_root:
        return "Error: No session workspace."
    root = Path(session_root).resolve()
    source_path = (root / source).resolve()
    dest_path = (root / dest).resolve()
    try:
        source_path.relative_to(root)
        dest_path.relative_to(root)
    except ValueError:
        return "Error: Both source and dest must be inside the session folder."
    if not source_path.exists():
        return f"Error: Source file '{source}' does not exist."
    if source_path.is_dir():
        return f"Error: '{source}' is a directory, not a file."
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(dest_path))
    return f"Moved '{source}' to '{dest}' in session folder"


def read_deliverable_impl(task_dir: str, task_name: str, filepath: str) -> str:
    """Read a file from the task's deliverables folder."""
    if not task_dir:
        return "Error: No active task. Call delegate_task first."
    deliverables_dir = Path(task_dir) / f"deliverables_{task_name}"
    path = (deliverables_dir / filepath).resolve()
    try:
        path.relative_to(deliverables_dir.resolve())
    except ValueError:
        return "Error: filepath must be inside the deliverables directory."
    if not path.exists():
        return f"Error: File '{filepath}' does not exist in deliverables."
    if path.is_dir():
        return f"Error: '{filepath}' is a directory, not a file."
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"


def update_persistent_memo_impl(project_dir: str, content: str) -> str:
    """Append an update to the .agent_memo.md file in the project directory."""
    memo_path = Path(project_dir) / ".agent_memo.md"
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        header = f"\n\n---\nMemo update at {timestamp}:\n\n"
        if memo_path.exists():
            existing = memo_path.read_text(encoding="utf-8")
        else:
            existing = "# Persistent Project Memo\n"
        new_text = f"{existing}{header}{content.strip()}\n"
        memo_path.write_text(new_text, encoding="utf-8")
        return f"Appended update to .agent_memo.md ({len(content)} bytes). This memo will be included in your system prompt on every turn."
    except Exception as e:
        return f"Error writing memo: {str(e)}"


def read_project_memo(project_dir: str) -> str:
    """Read the .agent_memo.md file from the project directory. Returns empty string if file doesn't exist."""
    memo_path = Path(project_dir) / ".agent_memo.md"
    if not memo_path.exists():
        return ""
    try:
        return memo_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading memo: {str(e)}"


# ---------------------------------------------------------------------------
# Main Agent tool definitions
# ---------------------------------------------------------------------------

def get_main_agent_tools(project_dir: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "Delegate a coding or testing task to the Sub-Agent. Creates a task folder. The Sub-Agent places deliverables in task_folder/deliverables_<task_name>/. IMPORTANT: Instruct the Sub-Agent to be concise - avoid overdoing test cases and error handling. tell it the files are in its current working directory. DO NOT pass absolute paths or mention the session root, as the Sub-Agent is securely jailed and will get access errors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instructions": {"type": "string", "description": "Clear instructions for the Sub-Agent. Be concise."},
                        "expected_deliverable": {"type": "string", "description": "What you expect back."},
                        "task_name": {"type": "string", "description": "Name for the task folder (e.g. 'integer_sorter', 'calculator')."},
                        "context_files": {"type": "array", "items": {"type": "string"}, "description": "Optional list of filenames from the session root to copy into the Sub-Agent's sandbox before it starts. CRITICAL: If the Sub-Agent needs to read, test, or modify existing files, you MUST provide this list of filenames from the session root. If you do not provide this, the Sub-Agent will start in a completely empty folder and fail."},
                    },
                    "required": ["instructions", "expected_deliverable", "task_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_sub_agent_task",
                "description": "Update the Sub-Agent's task instructions while it's running. Use this when user interventions indicate the task needs to change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_instructions": {"type": "string", "description": "Updated instructions for the Sub-Agent."},
                        "new_expected_deliverable": {"type": "string", "description": "Updated expected deliverable."},
                    },
                    "required": ["new_instructions", "new_expected_deliverable"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_deliverable",
                "description": "Read a file from a specific task's deliverables folder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_name": {"type": "string", "description": "The name of the task (e.g. 'calculator')."},
                        "filepath": {"type": "string", "description": "Path relative to deliverables folder (e.g. 'sorter.py')."},
                    },
                    "required": ["task_name", "filepath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_project_file",
                "description": "Write content to a file in the session folder (session root). Files written here are part of the session workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path relative to session folder root."},
                        "content": {"type": "string", "description": "Full file content to write."},
                    },
                    "required": ["filepath", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move_file",
                "description": "Move a file within the session folder (e.g. from task_<name>/deliverables_<name>/ to session root).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source file path relative to the session folder."},
                        "dest": {"type": "string", "description": "Destination file path relative to the session folder."},
                    },
                    "required": ["source", "dest"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_persistent_memo",
                "description": "Append an update to the persistent project memo (.agent_memo.md) in the project directory. This memo is automatically included in your system prompt on every turn.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string", "description": "The content to append to the persistent memo. It will be added as a new dated entry."}},
                    "required": ["content"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# JSON / debug utilities
# ---------------------------------------------------------------------------

class SafeEncoder(json.JSONEncoder):
    """Safely encodes SDK objects and unknown types to JSON without crashing."""
    def default(self, obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def debug_log_turn(debug_log_dir: str | None, agent: str, label: str, payload) -> None:
    """If debug_log_dir is set, write a JSON debug file for the given turn."""
    if not debug_log_dir:
        return
    try:
        root = Path(debug_log_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
        filename = root / f"{ts}_{agent}_{safe_label}.json"
        with filename.open("w", encoding="utf-8") as f:
            json.dump({"timestamp": ts, "agent": agent, "label": label, "payload": payload}, f, ensure_ascii=False, indent=2, cls=SafeEncoder)
    except Exception:
        return
