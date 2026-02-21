#!/usr/bin/env python3
"""
Coding Agent: Two-tier system with Main Agent (manager) and Sub-Agent (worker).
Main Agent delegates tasks to Sub-Agent running in a sandbox with async intervention support.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
import aioconsole
import typer
from openrouter import OpenRouter
from rich.console import Console
from rich.markdown import Markdown

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_FILENAMES = ("config.json",)
# Directory for config: same dir as this script, then cwd
_script_dir = Path(__file__).resolve().parent


def load_config() -> dict:
    """Load config from config.json (api_key, model, and other parameters). Sets OPENROUTER_API_KEY if not already set."""
    for base in (_script_dir, Path.cwd()):
        for name in CONFIG_FILENAMES:
            path = base / name
            if path.is_file():
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    api_key = data.get("api_key") or data.get("openrouter_api_key")
                    if api_key and "OPENROUTER_API_KEY" not in os.environ:
                        os.environ["OPENROUTER_API_KEY"] = api_key
                    return data
                except (json.JSONDecodeError, OSError):
                    pass
    return {}


_config = load_config()
DEFAULT_MODEL = _config.get("model", "openai/gpt-4o-mini")
SUB_AGENT_MAX_TURNS = _config.get("sub_agent_max_turns", 30)
SUB_AGENT_SLIDING_WINDOW = _config.get("sub_agent_sliding_window", 10)
SHELL_TIMEOUT = _config.get("shell_timeout", 15)
KEEP_SANDBOX = _config.get("keep_sandbox", False)
SANDBOX_BASE_DIR = _config.get("sandbox_base_dir")

app = typer.Typer(help="Coding agent: Main agent delegates tasks to a sandboxed Sub-Agent.")
console = Console()


# ---------------------------------------------------------------------------
# Phase 1: Sandbox & Sub-Agent Tools
# ---------------------------------------------------------------------------

def _sandbox_path(sandbox_dir: str, filename: str) -> Path:
    """Normalize filename to sandbox: prepend sandbox path, resolve relative path. Ensures path stays within sandbox."""
    if not filename or filename.strip() == "":
        raise ValueError("Filename cannot be empty")
    sandbox_root = Path(sandbox_dir).resolve()
    path = (sandbox_root / filename).resolve()
    # Security: ensure path is within sandbox
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
        # Ensure we're writing a file, not a directory
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
        return f"Error writing '{filename}': {str(e)}"


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


SUB_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the sandbox. Use a path relative to the sandbox (e.g. 'script.py' or 'src/main.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Path to the file relative to the sandbox."},
                },
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
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run in the sandbox."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move a file from sandbox to the deliverables folder (for handing over to Main Agent). Use this to move final deliverables without reading them into context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source file path relative to sandbox."},
                    "dest": {"type": "string", "description": "Destination file path relative to deliverables folder (e.g. 'sorter.py' or 'tests/test_sorter.py')."},
                },
                "required": ["source", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copy a file from sandbox to the deliverables folder (for handing over to Main Agent). Use this to copy final deliverables without reading them into context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source file path relative to sandbox."},
                    "dest": {"type": "string", "description": "Destination file path relative to deliverables folder (e.g. 'sorter.py' or 'tests/test_sorter.py')."},
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
                "properties": {
                    "filename": {"type": "string", "description": "File path relative to sandbox."},
                },
                "required": ["filename"],
            },
        },
    },
]


def move_file_impl(sandbox_dir: str, deliverables_dir: str, source: str, dest: str) -> str:
    """Move a file from sandbox to deliverables folder."""
    source_path = _sandbox_path(sandbox_dir, source)
    if not source_path.exists():
        return f"Error: Source file '{source}' does not exist."
    if source_path.is_dir():
        return f"Error: '{source}' is a directory, not a file."
    
    dest_path = Path(deliverables_dir) / dest
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    import shutil
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
    
    import shutil
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
# Phase 2 & 3: Sub-Agent Async Loop with Intervention
# ---------------------------------------------------------------------------

async def openrouter_completion(model: str, messages: list[dict], tools: list[dict] = None, tool_choice: str = "auto"):
    """Call OpenRouter API with async support."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Set it in config.json or environment.")
    
    async with OpenRouter(api_key=api_key) as client:
        response = await client.chat.send_async(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response


SUB_AGENT_SYSTEM = """You are a Sub-Agent working in a temporary sandbox. You MUST use tools to complete tasks.

Available tools:
- read_file(filename): read a file in the sandbox
- write_file(filename, content): write a file in the sandbox  
- run_shell_command(command): run a shell command in the sandbox (e.g. pytest, python script.py)
- move_file(source, dest): move a file from sandbox to deliverables folder (for handing over to Main Agent)
- copy_file(source, dest): copy a file from sandbox to deliverables folder (for handing over to Main Agent)
- delete_file(filename): delete a file from the sandbox

CRITICAL RULES:
1. You MUST use tools to perform actions. Do NOT just describe what you would do - actually call the tools!
2. ALWAYS read the "TASK TO COMPLETE" section in the user's message carefully - that is what you must build
3. Do NOT create different projects - create exactly what is requested in the task
4. BE CONCISE: Avoid overdoing test cases and error handling. Create what's requested, not excessive edge cases or verbose error messages.
5. To create files: use write_file() tool
6. To read files: use read_file() tool
7. To run commands/tests: use run_shell_command() tool
8. To hand over deliverables: use move_file() or copy_file() to move/copy final files to the deliverables folder
9. FAIL FAST PROTOCOL:
If you realize you cannot complete the task because you lack necessary context, missing files, or lack the capability, DO NOT hallucinate or guess. 
Instead, immediately stop and provide a final summary stating: "FAILED: I need [X] to complete this task." 
The Main Agent will read this, re-arrange the workspace, and provide you with what you need.

Work step by step:
1. Read the TASK TO COMPLETE section in the user message
2. Use write_file() to create the requested code files (be concise, avoid excessive error handling)
3. Use write_file() to create test files if needed (keep tests focused, avoid over-testing)
4. Use run_shell_command() to run tests (e.g. "pytest test_file.py" or "python script.py")
5. Read files with read_file() if you need to check what was written
6. Fix any errors by writing corrected files
7. When done, use move_file() or copy_file() to hand over final deliverables to the deliverables folder

When the task is complete and all files are written, tested, and moved to deliverables, provide a brief summary message."""


def sliding_window(messages: list[dict], system_role: str, keep_turns: int = SUB_AGENT_SLIDING_WINDOW) -> list[dict]:
    """Keep system message, initial task message, plus last keep_turns back-and-forth (assistant + user/tool) pairs."""
    out = []
    initial_user_msg = None
    system_msg = None
    
    # Extract system and initial user message (the task)
    for m in messages:
        if m.get("role") == "system":
            system_msg = m
        elif m.get("role") == "user" and initial_user_msg is None:
            # Keep the first user message (the task) - check if it looks like a task
            content = m.get("content", "")
            if "TASK TO COMPLETE" in content or "Task:" in content or len(content) > 100:
                initial_user_msg = m
    
    if system_msg:
        out.append(system_msg)
    if initial_user_msg:
        out.append(initial_user_msg)
    
    # Now process the rest (skip system and initial user message)
    rest = []
    found_initial = False
    for m in messages:
        if m.get("role") == "system":
            continue
        if m.get("role") == "user" and not found_initial:
            # Skip the initial task message (we already added it)
            content = m.get("content", "")
            if "TASK TO COMPLETE" in content or "Task:" in content or len(content) > 100:
                found_initial = True
                continue
        rest.append(m)
    
    # keep last keep_turns "turns" (each turn can be assistant + user/tool_calls/tool)
    turns = []
    i = 0
    while i < len(rest):
        if rest[i].get("role") == "assistant":
            turn = [rest[i]]
            i += 1
            while i < len(rest) and rest[i].get("role") in ("user", "tool"):
                turn.append(rest[i])
                i += 1
            turns.append(turn)
        else:
            i += 1
    for turn in turns[-keep_turns:]:
        out.extend(turn)
    return out


async def run_sub_agent(
    instructions: str,
    expected_deliverable: str,
    model: str,
    intervention_event: asyncio.Event,
    user_instruction_queue: list[str],
    task_name: str,
    task_dir: str,
    sub_agent_memory_ref: dict,  # Shared reference to Sub-Agent memory for Main Agent access
    main_agent_intervention_queue: asyncio.Queue,  # Queue for Main Agent to receive interventions
    verbose: bool = False,
    debug_log_dir: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Run Sub-Agent in a sandbox within a task folder. Task A = LLM + tools; Task B = listen for user input.
    Interventions go to Main Agent, not Sub-Agent directly.
    Returns (final deliverable, Sub-Agent memory snapshot) tuple.
    """
    # Create sandbox subdirectory within task folder (with suffix to distinguish)
    sandbox_dir = Path(task_dir) / f"sandbox_{task_name}"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    sandbox_dir_str = str(sandbox_dir)
    
    # Create deliverables folder in task directory (with suffix to distinguish)
    deliverables_dir = Path(task_dir) / f"deliverables_{task_name}"
    deliverables_dir.mkdir(parents=True, exist_ok=True)
    deliverables_dir_str = str(deliverables_dir)
    
    if verbose:
        console.print(f"[dim yellow]Sandbox: {sandbox_dir_str}[/dim yellow]")
        console.print(f"[dim yellow]Deliverables: {deliverables_dir_str}[/dim yellow]")
    else:
        console.print(f"[dim]Sandbox: {sandbox_dir_str}[/dim]")
    
    deliverable, memory_snapshot = await _run_sub_agent_in_sandbox(
        sandbox_dir_str,
        deliverables_dir_str,
        instructions,
        expected_deliverable,
        model,
        intervention_event,
        user_instruction_queue,
        sub_agent_memory_ref,
        main_agent_intervention_queue,
        verbose,
        debug_log_dir,
    )
    return deliverable, memory_snapshot


async def _run_sub_agent_in_sandbox(
    sandbox_dir: str,
    deliverables_dir: str,
    instructions: str,
    expected_deliverable: str,
    model: str,
    intervention_event: asyncio.Event,
    user_instruction_queue: list[str],
    sub_agent_memory_ref: dict,
    main_agent_intervention_queue: asyncio.Queue,
    verbose: bool = False,
    debug_log_dir: str | None = None,
) -> tuple[str, list[dict]]:
    """Internal function that runs Sub-Agent logic inside a sandbox directory.
    Returns (deliverable, Sub-Agent memory snapshot) tuple."""
    
    def build_initial_task_message(inst: str, deliverable: str) -> str:
        """Build initial task message - can be updated by Main Agent."""
        return f"""TASK TO COMPLETE:
{inst}

EXPECTED DELIVERABLE:
{deliverable}

INSTRUCTIONS:
- You are working in a sandbox directory within a task folder
- Use write_file() tool to create all required files in the sandbox
- Use run_shell_command() tool to test your code (e.g. pytest, python script.py)
- Use read_file() tool if you need to check existing files
- When files are ready, use move_file() or copy_file() to hand them over to the deliverables folder
- Complete the task step by step using tools
- BE CONCISE: Avoid overdoing test cases and error handling. Create what's requested, not excessive edge cases or verbose error messages.
- When finished, provide a brief summary of what was created

IMPORTANT: Read the TASK TO COMPLETE section above carefully. You must create exactly what is requested.
Start by using write_file() to create the required files. When done, use move_file() or copy_file() to hand over deliverables.
Keep code concise - avoid excessive test cases and verbose error handling."""

    # Create memory and store reference for Main Agent access
    initial_task_message = build_initial_task_message(instructions, expected_deliverable)
    memory: list[dict] = [
        {"role": "system", "content": SUB_AGENT_SYSTEM},
        {"role": "user", "content": initial_task_message},
    ]
    # Store memory reference for Main Agent
    sub_agent_memory_ref["memory"] = memory
    sub_agent_memory_ref["update_task"] = lambda new_inst, new_deliverable: memory.__setitem__(
        1, {"role": "user", "content": build_initial_task_message(new_inst, new_deliverable)}
    )

    async def listener():
        """Background task that listens for user input during Sub-Agent execution.
        Sends interventions to Main Agent, not Sub-Agent."""
        while True:
            try:
                line = await aioconsole.ainput()
                if not line or not line.strip():
                    continue
                intervention_text = line.strip()
                console.print(f"[bold yellow]→ User intervention: {intervention_text}[/bold yellow]")
                # Send to Main Agent, not Sub-Agent
                await main_agent_intervention_queue.put(intervention_text)
                intervention_event.set()  # Signal that intervention happened (Main Agent will handle)
            except (EOFError, asyncio.CancelledError):
                break

    async def sub_agent_loop() -> str:
        last_assistant_content: str = ""
        turn_count = 0
        while turn_count < SUB_AGENT_MAX_TURNS:
            if verbose:
                console.print(f"\n[bold magenta]━━━ Sub-Agent Turn {turn_count + 1} ━━━[/bold magenta]")
                console.print(f"[dim]Memory before processing: {len(memory)} messages[/dim]")
            
            
            # Check if Main Agent updated the task instructions
            if "task_updated" in sub_agent_memory_ref:
                sub_agent_memory_ref.pop("task_updated")
                if verbose:
                    console.print("[bold yellow][Sub-Agent] ⚠ Task instructions updated by Main Agent[/bold yellow]")
                    console.print(f"[dim]Memory before sliding window: {len(memory)} messages[/dim]")
                    # Show what the updated task message looks like
                    if len(memory) > 1 and memory[1].get("role") == "user":
                        updated_task = memory[1].get("content", "")
                        console.print(f"[yellow]Updated task (memory[1]):[/yellow]")
                        task_lines = updated_task.split('\n')[:5]
                        for line in task_lines:
                            console.print(f"[yellow]  {line}[/yellow]")
                        if len(updated_task.split('\n')) > 5:
                            console.print(f"[yellow]  ...[/yellow]")
                
                # Re-slide window to ensure updated task is included
                memory_before = len(memory)
                memory[:] = sliding_window(memory, SUB_AGENT_SYSTEM)
                memory_after = len(memory)
                
                if verbose:
                    console.print(f"[dim]Memory after sliding window: {memory_before} → {memory_after} messages[/dim]")
                    # Verify updated task is still present
                    has_task = any(
                        m.get("role") == "user" and "TASK TO COMPLETE" in m.get("content", "")
                        for m in memory
                    )
                    if has_task:
                        console.print(f"[green]✓ Updated task preserved in memory[/green]")
                    else:
                        console.print(f"[red]✗ WARNING: Updated task NOT found in memory after sliding window![/red]")
            
            # Check if Main Agent wants to inject a user message
            if "inject_message" in sub_agent_memory_ref:
                inject_msg = sub_agent_memory_ref.pop("inject_message")
                memory.append({"role": "user", "content": inject_msg})
                if verbose:
                    console.print(f"[dim cyan][Sub-Agent] Main Agent injected message: {inject_msg[:100]}...[/dim cyan]")
                memory[:] = sliding_window(memory, SUB_AGENT_SYSTEM)

            # Trim to sliding window before each LLM call
            # Note: If intervention happened above, the user message is now in memory
            # and will be included in the next LLM call
            current_messages = sliding_window(memory, SUB_AGENT_SYSTEM)
            
            # CRITICAL: Always ensure initial task is included in every LLM call
            if current_messages and current_messages[0].get("role") == "system":
                # Check if initial task message is present
                has_initial_task = any(
                    m.get("role") == "user" and "TASK TO COMPLETE" in m.get("content", "")
                    for m in current_messages
                )
                if not has_initial_task:
                    # Re-add initial task message - it must always be present
                    if len(memory) > 1 and memory[1].get("role") == "user":
                        # Insert after system message
                        current_messages.insert(1, memory[1])
                        if verbose:
                            console.print("[dim yellow][Sub-Agent] Re-inserted initial task message[/dim yellow]")

            debug_log_turn(
                debug_log_dir,
                agent="sub",
                label=f"turn_{turn_count + 1}_before_llm",
                payload={"current_messages": current_messages},
            )

            if verbose:
                console.print(f"\n[bold cyan][Sub-Agent] Calling LLM (turn {turn_count + 1})[/bold cyan]")
                console.print(f"[dim]Model: {model}[/dim]")
                console.print(f"[dim]Total messages in memory: {len(current_messages)}[/dim]")
                console.print(f"[dim]Memory contents:[/dim]")
                for i, m in enumerate(current_messages):
                    role = m.get("role", "unknown")
                    content = m.get("content", "")
                    tool_calls = m.get("tool_calls")
                    
                    if role == "system":
                        console.print(f"[dim]  [{i}] system: {content[:100]}...[/dim]" if len(content) > 100 else f"[dim]  [{i}] system: {content}[/dim]")
                    elif role == "user":
                        if "TASK TO COMPLETE" in content:
                            # Show full task message
                            lines = content.split('\n')
                            console.print(f"[bold yellow]  [{i}] user (TASK):[/bold yellow]")
                            for line in lines[:10]:  # First 10 lines
                                console.print(f"[yellow]    {line}[/yellow]")
                            if len(lines) > 10:
                                console.print(f"[yellow]    ... ({len(lines) - 10} more lines)[/yellow]")
                        else:
                            preview = content[:200] + "..." if len(content) > 200 else content
                            console.print(f"[yellow]  [{i}] user: {preview}[/yellow]")
                    elif role == "assistant":
                        if tool_calls:
                            console.print(f"[dim cyan]  [{i}] assistant: [tool calls: {len(tool_calls)}][/dim cyan]")
                        else:
                            preview = content[:200] + "..." if len(content) > 200 else content
                            console.print(f"[cyan]  [{i}] assistant: {preview}[/cyan]")
                    elif role == "tool":
                        tool_result = content[:150] + "..." if len(content) > 150 else content
                        console.print(f"[dim green]  [{i}] tool: {tool_result}[/dim green]")

            response = await openrouter_completion(
                model=model,
                messages=current_messages,
                tools=SUB_AGENT_TOOLS,
                tool_choice="auto",
            )
            choice = response.choices[0] if response.choices else None
            if not choice or not choice.message:
                last_assistant_content = "No response from model."
                break

            msg = choice.message
            content = getattr(msg, "content", None) or ""
            if content:
                last_assistant_content = content
                if verbose:
                    console.print(f"[dim cyan][Sub-Agent] Response:[/dim cyan] {content[:200]}..." if len(content) > 200 else f"[dim cyan][Sub-Agent] Response:[/dim cyan] {content}")

            memory.append({"role": "assistant", "content": content or "", "tool_calls": getattr(msg, "tool_calls", None)})

            tool_calls = getattr(msg, "tool_calls", None) or []
            
            # --- ADD THIS BLOCK ---
            debug_log_turn(
                debug_log_dir,
                agent="sub",
                label=f"turn_{turn_count + 1}_after_llm",
                payload={
                    "content": content,
                    "tool_calls": str(tool_calls)
                },
            )
            # ----------------------
            
            if not tool_calls:
                # No tools called: check if Sub-Agent says it's done
                content_lower = content.lower()
                completion_keywords = ["done", "finished", "complete", "completed", "deliverable", "task complete"]
                is_complete = any(keyword in content_lower for keyword in completion_keywords)
                
                if is_complete and content.strip():
                    # Sub-Agent says it's done and provided a message
                    if verbose:
                        console.print("[dim cyan][Sub-Agent] Reported completion, finishing.[/dim cyan]")
                    break
                elif content.strip():
                    # Sub-Agent responded with text but didn't use tools - prompt to continue
                    if verbose:
                        console.print("[dim yellow][Sub-Agent] Responded with text but no tools. Prompting to use tools...[/dim yellow]")
                    memory.append({
                        "role": "user",
                        "content": "Please use the available tools (write_file, read_file, run_shell_command, move_file, copy_file) to complete the task. Do not just describe actions - actually execute them using tools."
                    })
                    turn_count += 1
                    continue
                else:
                    # Empty response - break
                    if verbose:
                        console.print("[dim cyan][Sub-Agent] Empty response, finishing.[/dim cyan]")
                    break

            if verbose:
                console.print(f"[dim cyan][Sub-Agent] Executing {len(tool_calls)} tool call(s)[/dim cyan]")

            for tc in tool_calls:
                # Check if Main Agent updated instructions during tool execution
                if "task_updated" in sub_agent_memory_ref or "inject_message" in sub_agent_memory_ref:
                    if verbose:
                        console.print("[dim yellow][Sub-Agent] Main Agent updated instructions, stopping tool execution...[/dim yellow]")
                    break  # stop current tool execution, continue loop to re-call LLM with updated context
                tid = getattr(tc, "id", None) or ""
                f = getattr(tc, "function", None)
                if not f:
                    continue
                fname = getattr(f, "name", "") or ""
                fargs_str = getattr(f, "arguments", "{}") or "{}"
                try:
                    fargs = json.loads(fargs_str)
                except json.JSONDecodeError:
                    fargs = {}
                if verbose:
                    console.print(f"[dim cyan][Sub-Agent] Tool: {fname}({json.dumps(fargs, indent=2)})[/dim cyan]")
                result = execute_sub_agent_tool(sandbox_dir, deliverables_dir, fname, fargs)
                if verbose:
                    result_preview = result[:200] + "..." if len(result) > 200 else result
                    console.print(f"[dim green][Sub-Agent] Tool result:[/dim green] {result_preview}")
                memory.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": result,
                })

            turn_count += 1

        return last_assistant_content or "(Sub-Agent finished with no final message.)"

    listener_task = asyncio.create_task(listener())
    try:
        deliverable = await sub_agent_loop()
        # Return memory snapshot for Main Agent visibility
        memory_snapshot = memory.copy()
        return deliverable, memory_snapshot
    finally:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Phase 4 & Main Agent: delegate_task and Memory
# ---------------------------------------------------------------------------

MAIN_AGENT_SYSTEM_BASE = """You are the Main Agent (Manager). You interact with the user and manage a coding project.
You have six tools:
1. delegate_task(instructions, expected_deliverable, task_name, context_files?) - hand off coding/testing to a Sub-Agent in a sandbox. Creates session_workspace/task_<name>/. Optional context_files: list of filenames from the session root to copy into the sandbox before the Sub-Agent starts (e.g. so it can test or modify main.py, generator.py).
2. update_sub_agent_task(new_instructions, new_expected_deliverable) - update the Sub-Agent's task instructions while it's running. Use this when user interventions indicate the task needs to change.
3. read_deliverable(task_name, filepath) - read a file from a specific task's deliverables folder (session_workspace/task_<name>/deliverables_<name>/).
4. write_project_file(filepath, content) - write a file into the session folder root. Files written here are part of the session workspace.
5. move_file(source, dest) - move a file within the session folder (e.g. from task_<name>/deliverables_<name>/ to session root).
6. update_persistent_memo(content) - append an update to the persistent project memo (.agent_memo.md). This memo is automatically included in your system prompt on every turn, allowing you to maintain persistent memory across sessions.

IMPORTANT: BE CONCISE. When delegating tasks, instruct the Sub-Agent to avoid overdoing test cases and error handling. Keep code focused and practical.

INTERVENTIONS: If the user intervenes during Sub-Agent execution (types a message), you will receive those interventions IMMEDIATELY. You should:
- Process the intervention and understand what the user wants
- Use update_persistent_memo() to record the user's intervention and any changes to the task. Do this only when the current turn is user intervention.
- If the task needs to change, use update_sub_agent_task() to update the Sub-Agent's instructions
- If you need to provide additional guidance, you can inject messages into Sub-Agent's context
- You have visibility into Sub-Agent's memory - use this to understand what it's working on

PROJECT MEMO: The contents of .agent_memo.md (if it exists) are included below. Use update_persistent_memo() to record important decisions, project structure, key learnings, or any information you want to remember across sessions.

Workflow:
- You operate out of a master Session Workspace folder.
- Call delegate_task() with a task_name. This creates a subfolder: session_workspace/task_<name>/.
- When a Sub-Agent finishes, it leaves files in session_workspace/task_<name>/deliverables_<name>/.
- ALWAYS use move_file() to pull the final files out of the task's deliverables folder and into your main session root.
- Use read_deliverable(task_name, filepath) to read files from a specific task's deliverables folder.
- Use write_project_file() to write files to the session root.
- Use update_persistent_memo() to record important information for future reference.
- Report clearly what was created. Be concise."""


def build_main_agent_system(memo_content: str) -> str:
    """Build Main Agent system prompt with memo content injected."""
    if memo_content.strip():
        return f"""{MAIN_AGENT_SYSTEM_BASE}

---
PROJECT MEMO (.agent_memo.md):
{memo_content}
---
"""
    return MAIN_AGENT_SYSTEM_BASE


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
    import shutil
    shutil.move(str(source_path), str(dest_path))
    return f"Moved '{source}' to '{dest}' in session folder"


def get_main_agent_tools(project_dir: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "Delegate a coding or testing task to the Sub-Agent. Creates a task folder. The Sub-Agent places deliverables in task_folder/deliverables_<task_name>/. IMPORTANT: Instruct the Sub-Agent to be concise - avoid overdoing test cases and error handling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instructions": {"type": "string", "description": "Clear instructions for the Sub-Agent (e.g. 'Write a Python script that sorts a list of dicts by key, and write pytest tests'). Be concise - avoid requesting excessive test cases or verbose error handling."},
                        "expected_deliverable": {"type": "string", "description": "What you expect back (e.g. 'The script code and passing pytest output')."},
                        "task_name": {"type": "string", "description": "Name for the task folder (e.g. 'integer_sorter', 'calculator'). Will create task_<name> folder."},
                        "context_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of filenames from the session root to copy into the Sub-Agent's sandbox before it starts (e.g. ['generator.py', 'main.py']). Use this if the Sub-Agent needs to test, read, or modify existing files.",
                        },
                    },
                    "required": ["instructions", "expected_deliverable", "task_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_sub_agent_task",
                "description": "Update the Sub-Agent's task instructions while it's running. Use this when user interventions indicate the task needs to change. The Sub-Agent will see the updated instructions on its next turn.",
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
                        "filepath": {"type": "string", "description": "Path relative to session folder root (e.g. 'sorter.py', 'tests/test_sorter.py')."},
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
                        "source": {"type": "string", "description": "Source file path relative to the session folder (e.g. 'task_sorter/deliverables_sorter/sorter.py')."},
                        "dest": {"type": "string", "description": "Destination file path relative to the session folder (e.g. 'sorter.py' to put it in the session root)."},
                    },
                    "required": ["source", "dest"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_persistent_memo",
                "description": "Append an update to the persistent project memo (.agent_memo.md) in the project directory. This memo is automatically included in your system prompt on every turn, allowing you to maintain persistent memory across sessions. Use this to record important decisions, project structure, key learnings, or any information you want to remember.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The content to append to the persistent memo. It will be added as a new dated entry."},
                    },
                    "required": ["content"],
                },
            },
        },
    ]


def read_deliverable_impl(task_dir: str, task_name: str, filepath: str) -> str:
    """Read a file from the task's deliverables folder."""
    if not task_dir:
        return "Error: No active task. Call delegate_task first."
    deliverables_dir = Path(task_dir) / f"deliverables_{task_name}"
    path = (deliverables_dir / filepath).resolve()
    try:
        path.relative_to(deliverables_dir.resolve())
    except ValueError:
        return f"Error: filepath must be inside the deliverables directory."
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
    
    
class SafeEncoder(json.JSONEncoder):
    """Safely encodes SDK objects and unknown types to JSON without crashing."""
    def default(self, obj):
        # Handle modern Pydantic/SDK objects
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # Handle older SDK objects
        if hasattr(obj, "dict"):
            return obj.dict()
        # If all else fails, force it into a string
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def debug_log_turn(debug_log_dir: str | None, agent: str, label: str, payload) -> None:
    """
    If debug_log_dir is set, write a JSON debug file for the given turn.
    Used to trace where agent behavior deviates from expectations.
    """
    if not debug_log_dir:
        return
    try:
        root = Path(debug_log_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
        filename = root / f"{ts}_{agent}_{safe_label}.json"
        with filename.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": ts,
                    "agent": agent,
                    "label": label,
                    "payload": payload,
                },
                f,
                ensure_ascii=False,
                indent=2,
                cls=SafeEncoder,  # <--- ADD THE ENCODER HERE
            )
    except Exception:
        # Debug logging must never break the agent
        return


def main_agent_sliding_window(messages: list[dict], keep_turns: int = 10) -> list[dict]:
    """Sliding window for Main Agent: always keeps system message and initial user prompt, then last keep_turns turns."""
    if not messages:
        return []
    
    out = []
    system_msg = None
    initial_user_msg = None
    
    # Extract system message (first message with role="system")
    for m in messages:
        if m.get("role") == "system":
            system_msg = m
            break
    
    # Extract initial user prompt (first user message after system)
    found_system = False
    for m in messages:
        if m.get("role") == "system":
            found_system = True
            continue
        if found_system and m.get("role") == "user" and initial_user_msg is None:
            initial_user_msg = m
            break
    
    # Always include system message
    if system_msg:
        out.append(system_msg)
    
    # Always include initial user prompt
    if initial_user_msg:
        out.append(initial_user_msg)
    
    # Now process the rest (skip system and initial user message)
    rest = []
    found_system = False
    found_initial_user = False
    for m in messages:
        if m.get("role") == "system":
            found_system = True
            continue
        if found_system and not found_initial_user and m.get("role") == "user":
            found_initial_user = True
            continue
        if found_system and found_initial_user:
            rest.append(m)
    
    # Keep last keep_turns "turns" (each turn can be assistant + user/tool_calls/tool)
    turns = []
    i = 0
    while i < len(rest):
        if rest[i].get("role") == "assistant":
            turn = [rest[i]]
            i += 1
            while i < len(rest) and rest[i].get("role") in ("user", "tool"):
                turn.append(rest[i])
                i += 1
            turns.append(turn)
        elif rest[i].get("role") == "user":
            # Standalone user message (not part of assistant turn)
            turn = [rest[i]]
            i += 1
            while i < len(rest) and rest[i].get("role") == "tool":
                turn.append(rest[i])
                i += 1
            turns.append(turn)
        else:
            i += 1
    
    # Add last keep_turns turns
    for turn in turns[-keep_turns:]:
        out.extend(turn)
    
    return out


async def run_main_agent_loop(
    initial_prompt: str,
    model: str,
    cwd: str,
    verbose: bool = False,
    debug_log_dir: str | None = None,
) -> None:
    """Main Agent Loop with Session Hierarchy."""
    # --- 1. Create Session Folder ---
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(cwd) / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)
    session_dir_str = str(session_dir)

    if verbose:
        console.print(f"[bold green]Session Workspace created:[/bold green] {session_dir_str}")

    # --- 2. Initialize/Update Persistent Memo (Stays in cwd) ---
    memo_path = Path(cwd) / ".agent_memo.md"
    if not memo_path.exists():
        update_persistent_memo_impl(cwd, f"**Project Initialized.**\nInitial Goal:\n{initial_prompt}")
    else:
        update_persistent_memo_impl(cwd, f"**New Session Started.**\nWorkspace: session_{session_id}\nPrompt:\n{initial_prompt}")

    # Read initial memo
    initial_memo = read_project_memo(cwd)
    initial_system = build_main_agent_system(initial_memo)
    
    memory: list[dict] = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_prompt},
    ]
    intervention_event = asyncio.Event()
    user_instruction_queue: list[str] = []
    # Shared reference to Sub-Agent memory (for Main Agent to access/modify)
    sub_agent_memory_ref: dict = {}
    # Queue for Main Agent to receive interventions immediately
    main_agent_intervention_queue: asyncio.Queue = asyncio.Queue()

    while True:
        # Read memo and rebuild system prompt dynamically
        current_memo = read_project_memo(cwd)
        current_system = build_main_agent_system(current_memo)
        
        # Update system message in memory (always first message)
        if memory and memory[0].get("role") == "system":
            memory[0]["content"] = current_system
        else:
            # Shouldn't happen, but handle gracefully
            memory.insert(0, {"role": "system", "content": current_system})
        
        # Apply sliding window (preserves system + initial prompt + last ~10 turns)
        memory = main_agent_sliding_window(memory, keep_turns=10)

        # Debug-log this Main Agent turn if requested
        debug_log_turn(
            debug_log_dir,
            agent="main",
            label="before_llm_call",
            payload={"memory": memory},
        )
        
        with console.status("[bold green]Main Agent thinking...", spinner="dots"):
            if verbose:
                console.print(f"\n[bold cyan][Main Agent] Calling LLM[/bold cyan]")
                console.print(f"[dim]Model: {model}[/dim]")
                console.print(f"[dim]Messages after sliding window: {len(memory)}[/dim]")
                if current_memo.strip():
                    console.print(f"[dim]Memo included: {len(current_memo)} bytes[/dim]")
            response = await openrouter_completion(
                model=model,
                messages=memory,
                tools=get_main_agent_tools(cwd),
                tool_choice="auto",
            )

        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message:
            console.print("[red]No response from model.[/red]")
            break

        msg = choice.message
        content = getattr(msg, "content", None) or ""
        if content:
            console.print(Markdown(content))
            if verbose:
                console.print(f"[dim cyan][Main Agent] Full response length: {len(content)} chars[/dim cyan]")

        tool_calls = getattr(msg, "tool_calls", None) or []
        
        # --- ADD THIS BLOCK ---
        debug_log_turn(
            debug_log_dir,
            agent="main",
            label="after_llm_call",
            payload={
                "content": content,
                "tool_calls": str(tool_calls) # Cast to string to prevent JSON serialization crashes
            },
        )
        
        if not tool_calls:
            # No tools: ask user for next input or break
            if verbose:
                console.print("[dim cyan][Main Agent] No tool calls, waiting for user input[/dim cyan]")
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: console.input("\n[bold blue]You:[/bold blue] "))
                if not user_input.strip():
                    continue
                if user_input.strip().lower() in ("exit", "quit", "q"):
                    break
                if verbose:
                    console.print(f"[dim cyan][Main Agent] User input: {user_input}[/dim cyan]")
                memory.append({"role": "assistant", "content": content or ""})
                memory.append({"role": "user", "content": user_input})
                continue
            except EOFError:
                break

        memory.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})

        if verbose:
            console.print(f"[bold cyan][Main Agent] Executing {len(tool_calls)} tool call(s)[/bold cyan]")

        for tc in tool_calls:
            tid = getattr(tc, "id", None) or ""
            f = getattr(tc, "function", None)
            if not f:
                continue
            fname = getattr(f, "name", "") or ""
            fargs_str = getattr(f, "arguments", "{}") or "{}"
            try:
                fargs = json.loads(fargs_str)
            except json.JSONDecodeError:
                fargs = {}

            if verbose:
                console.print(f"[bold cyan][Main Agent] Tool: {fname}({json.dumps(fargs, indent=2)})[/bold cyan]")

            if fname == "delegate_task":
                instructions = fargs.get("instructions", "")
                expected_deliverable = fargs.get("expected_deliverable", "Completed task.")
                import time
                task_name = fargs.get("task_name", f"task_{int(time.time())}")
                context_files = fargs.get("context_files", [])

                instructions_with_conciseness = f"{instructions}\n\nIMPORTANT: Be concise. Avoid overdoing test cases and error handling."

                # Create task folder inside the session folder
                task_dir = Path(session_dir_str) / f"task_{task_name}"
                task_dir.mkdir(exist_ok=True)

                # Pre-create sandbox and copy context files from session root
                sandbox_dir = task_dir / f"sandbox_{task_name}"
                sandbox_dir.mkdir(parents=True, exist_ok=True)
                if context_files and isinstance(context_files, list):
                    import shutil
                    copied = []
                    for filename in context_files:
                        if not isinstance(filename, str) or not filename.strip():
                            continue
                        src_path = Path(session_dir_str) / filename
                        if src_path.is_file():
                            dst_path = sandbox_dir / filename
                            dst_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_path, dst_path)
                            copied.append(filename)
                    if verbose and copied:
                        console.print(f"[dim green][Main Agent] Copied to sandbox: {', '.join(copied)}[/dim green]")

                if verbose:
                    console.print(f"[bold cyan][Main Agent] Delegating task:[/bold cyan] {instructions[:100]}...")
                    console.print(f"[dim]Task folder: {task_dir}[/dim]")
                else:
                    console.print(f"[dim]Task folder: {task_dir}[/dim]")
                
                # Clear memory reference for new task
                sub_agent_memory_ref.clear()
                
                # Run Sub-Agent with concurrent intervention processing
                sub_agent_running = True
                interventions_received_during_execution: list[str] = []
                
                async def process_interventions():
                    """Process interventions immediately: add to memory and let Main Agent respond."""
                    nonlocal sub_agent_running, interventions_received_during_execution
                    while sub_agent_running:
                        try:
                            intervention = await asyncio.wait_for(
                                main_agent_intervention_queue.get(), timeout=0.1
                            )
                            interventions_received_during_execution.append(intervention)
                            console.print(f"[bold cyan][Main Agent] Received intervention: {intervention}[/bold cyan]")
                            
                            # Add intervention to Main Agent memory
                            memory.append({
                                "role": "user",
                                "content": (
                                    f"[User intervention during Sub-Agent execution] {intervention}\n\n"
                                    f"*** SYSTEM INSTRUCTION ***\n"
                                    f"1. If this changes the task, use `update_sub_agent_task` to redirect the worker.\n"
                                    f"2. You MUST use `update_persistent_memo` to record this new requirement so it is not forgotten."
                                )
                            })
                            # Immediately process intervention: call Main Agent LLM to decide what to do
                            # This allows Main Agent to call update_sub_agent_task() while Sub-Agent is running
                            try:
                                if verbose:
                                    console.print(f"[bold cyan][Main Agent] Processing intervention immediately...[/bold cyan]")
                                
                                # Read memo and rebuild system prompt
                                current_memo = read_project_memo(cwd)
                                current_system = build_main_agent_system(current_memo)
                                
                                # Update system message in memory
                                intervention_memory = memory.copy()
                                if intervention_memory and intervention_memory[0].get("role") == "system":
                                    intervention_memory[0]["content"] = current_system
                                
                                # Apply sliding window
                                intervention_memory = main_agent_sliding_window(intervention_memory, keep_turns=10)

                                # Debug-log this intervention turn if requested
                                debug_log_turn(
                                    debug_log_dir,
                                    agent="main",
                                    label="intervention_before_llm_call",
                                    payload={"memory": intervention_memory},
                                )
                                
                                # Call Main Agent LLM with current memory (including intervention)
                                with console.status("[bold cyan]Main Agent processing intervention...", spinner="dots"):
                                    response = await openrouter_completion(
                                        model=model,
                                        messages=intervention_memory,
                                        tools=get_main_agent_tools(cwd),
                                        tool_choice="auto",
                                    )
                                
                                choice = response.choices[0] if response.choices else None
                                if choice and choice.message:
                                    msg = choice.message
                                    content = getattr(msg, "content", None) or ""
                                    if content:
                                        console.print(Markdown(content))
                                    
                                    tool_calls = getattr(msg, "tool_calls", None) or []
                                    if tool_calls:
                                        # Main Agent wants to call tools (e.g., update_sub_agent_task)
                                        memory.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})
                                        
                                        for tc in tool_calls:
                                            tid = getattr(tc, "id", None) or ""
                                            f = getattr(tc, "function", None)
                                            if not f:
                                                continue
                                            fname = getattr(f, "name", "") or ""
                                            fargs_str = getattr(f, "arguments", "{}") or "{}"
                                            try:
                                                fargs = json.loads(fargs_str)
                                            except json.JSONDecodeError:
                                                fargs = {}
                                            
                                            if verbose:
                                                console.print(f"[bold cyan][Main Agent] Tool (intervention): {fname}({json.dumps(fargs, indent=2)})[/bold cyan]")
                                            
                                            # Handle update_sub_agent_task tool call
                                            if fname == "update_sub_agent_task":
                                                new_instructions = fargs.get("new_instructions", "")
                                                new_expected_deliverable = fargs.get("new_expected_deliverable", "")
                                                if not new_instructions or not new_expected_deliverable:
                                                    result = "Error: 'new_instructions' and 'new_expected_deliverable' are required."
                                                elif "update_task" not in sub_agent_memory_ref:
                                                    result = "Error: No active Sub-Agent task to update. Call delegate_task first."
                                                else:
                                                    # Update Sub-Agent's task instructions
                                                    sub_agent_memory_ref["update_task"](new_instructions, new_expected_deliverable)
                                                    sub_agent_memory_ref["task_updated"] = True
                                                    result = f"Updated Sub-Agent task instructions. New instructions: {new_instructions[:100]}..."
                                                    if verbose:
                                                        console.print(f"[bold cyan][Main Agent] Updated Sub-Agent task:[/bold cyan] {new_instructions[:100]}...")
                                                memory.append({"role": "tool", "tool_call_id": tid, "content": result})
                                            elif fname == "update_persistent_memo":
                                                content = fargs.get("content", "")
                                                if not content:
                                                    result = "Error: 'content' argument is required for update_persistent_memo."
                                                else:
                                                    result = update_persistent_memo_impl(cwd, content)
                                                if verbose:
                                                    console.print(f"[bold cyan][Main Agent] Updated persistent memo:[/bold cyan] {len(content)} bytes")
                                                memory.append({"role": "tool", "tool_call_id": tid, "content": result})
                                            # Other tools can be handled here if needed
                                    
                                    else:
                                        # No tools, just add response to memory
                                        memory.append({"role": "assistant", "content": content or ""})
                            except Exception as e:
                                if verbose:
                                    console.print(f"[dim red][Main Agent] Error processing intervention: {e}[/dim red]")
                            
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break
                
                async def run_sub_agent_task():
                    """Run Sub-Agent and return result."""
                    return await run_sub_agent(
                        instructions=instructions_with_conciseness,
                        expected_deliverable=expected_deliverable,
                        model=model,
                        intervention_event=intervention_event,
                        user_instruction_queue=user_instruction_queue,
                        task_name=task_name,
                        task_dir=str(task_dir),
                        sub_agent_memory_ref=sub_agent_memory_ref,
                        main_agent_intervention_queue=main_agent_intervention_queue,
                        verbose=verbose,
                        debug_log_dir=debug_log_dir,
                    )
                
                # Run Sub-Agent and process interventions concurrently
                with console.status("[bold yellow]Sub-Agent working in sandbox... (type a message and Enter to intervene)[/bold yellow]", spinner="dots"):
                    intervention_processor_task = asyncio.create_task(process_interventions())
                    try:
                        deliverable, sub_agent_memory_snapshot = await run_sub_agent_task()
                    finally:
                        sub_agent_running = False
                        # Collect any remaining interventions
                        while not main_agent_intervention_queue.empty():
                            try:
                                interv = main_agent_intervention_queue.get_nowait()
                                if interv not in interventions_received_during_execution:
                                    interventions_received_during_execution.append(interv)
                            except asyncio.QueueEmpty:
                                break
                        intervention_processor_task.cancel()
                        try:
                            await intervention_processor_task
                        except asyncio.CancelledError:
                            pass
                
                # Process interventions that arrived during Sub-Agent execution
                # (They were already added to memory by monitor_interventions, but we note them here)
                if interventions_received_during_execution:
                    console.print(f"[bold yellow]Interventions received during Sub-Agent execution: {len(interventions_received_during_execution)}[/bold yellow]")
                
                # Add Sub-Agent memory visibility to Main Agent memory
                if sub_agent_memory_snapshot and len(sub_agent_memory_snapshot) > 0:
                    memory.append({
                        "role": "system",
                        "content": f"[Sub-Agent Memory Snapshot] Last {min(5, len(sub_agent_memory_snapshot))} messages from Sub-Agent:\n" + 
                                   "\n".join([
                                       f"{m.get('role', 'unknown')}: {str(m.get('content', ''))[:200]}..." 
                                       if len(str(m.get('content', ''))) > 200 
                                       else f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                                       for m in sub_agent_memory_snapshot[-5:]
                                   ])
                    })
                
                #summary = f"Sub-Agent completed. Task folder: {task_dir}\nDeliverables are in {task_dir}/deliverables_{task_name}/\nSummary: {deliverable}"
                summary = (
                    f"Sub-Agent completed. Task folder: {task_dir}\nDeliverables are in {task_dir}/deliverables_{task_name}/\nSummary: {deliverable}"
                    f"*** SYSTEM INSTRUCTION ***\n"
                    f"1. Evaluate the Sub-Agent's summary. Did it succeed, or did it fail/ask for more context?\n"
                    f"2. If it SUCCEEDED, use `move_file` to move the final files from the sandbox into the session root.\n"
                    f"3. If it FAILED, figure out what it needs (e.g., use `delegate_task` again and pass the missing files using `context_files`).\n"
                    f"4. You MUST use `update_persistent_memo` EXACTLY ONCE to record the outcome (success or failure) of this specific task."
                )
                if interventions_received_during_execution:
                    summary += f"\n\nNote: {len(interventions_received_during_execution)} user intervention(s) were received during execution and added to Main Agent memory."
                if verbose:
                    console.print(f"[bold cyan][Main Agent] Received deliverable:[/bold cyan] {deliverable[:200]}..." if len(deliverable) > 200 else f"[bold cyan][Main Agent] Received deliverable:[/bold cyan] {deliverable}")
                memory.append({"role": "tool", "tool_call_id": tid, "content": summary})
            elif fname == "update_sub_agent_task":
                new_instructions = fargs.get("new_instructions", "")
                new_expected_deliverable = fargs.get("new_expected_deliverable", "")
                if not new_instructions or not new_expected_deliverable:
                    result = "Error: 'new_instructions' and 'new_expected_deliverable' are required."
                elif "update_task" not in sub_agent_memory_ref:
                    result = "Error: No active Sub-Agent task to update. Call delegate_task first."
                else:
                    # Update Sub-Agent's task instructions
                    if verbose:
                        console.print(f"[bold cyan][Main Agent] Updating Sub-Agent task...[/bold cyan]")
                        console.print(f"[cyan]  Old task (from memory):[/cyan]")
                        if "memory" in sub_agent_memory_ref and len(sub_agent_memory_ref["memory"]) > 1:
                            old_task = sub_agent_memory_ref["memory"][1].get("content", "")[:300]
                            console.print(f"[dim]    {old_task}...[/dim]")
                        console.print(f"[cyan]  New instructions:[/cyan]")
                        console.print(f"[yellow]    {new_instructions[:300]}...[/yellow]")
                    
                    sub_agent_memory_ref["update_task"](new_instructions, new_expected_deliverable)
                    sub_agent_memory_ref["task_updated"] = True
                    
                    if verbose:
                        console.print(f"[bold green]✓ Sub-Agent task updated. Memory now contains:[/bold green]")
                        if "memory" in sub_agent_memory_ref:
                            updated_task = sub_agent_memory_ref["memory"][1].get("content", "")[:300]
                            console.print(f"[green]    {updated_task}...[/green]")
                    
                    result = f"Updated Sub-Agent task instructions. New instructions: {new_instructions[:100]}..."
                memory.append({"role": "tool", "tool_call_id": tid, "content": result})
            elif fname == "read_deliverable":
                filepath = fargs.get("filepath", "")
                task_name = fargs.get("task_name", "")
                if not filepath or not task_name:
                    result = "Error: 'filepath' and 'task_name' arguments are required."
                else:
                    result = read_deliverable_impl(str(Path(session_dir_str) / f"task_{task_name}"), task_name, filepath)
                if verbose:
                    console.print(f"[bold cyan][Main Agent] Read deliverable:[/bold cyan] {filepath} ({len(result)} chars)")
                memory.append({"role": "tool", "tool_call_id": tid, "content": result})
            elif fname == "write_project_file":
                filepath = fargs.get("filepath", "")
                content = fargs.get("content", "")
                if verbose:
                    console.print(f"[bold cyan][Main Agent] Writing file:[/bold cyan] {filepath} ({len(content)} bytes)")
                result = write_project_file_impl(session_dir_str, filepath, content)
                if verbose:
                    console.print(f"[dim green][Main Agent] File write result:[/dim green] {result}")
                memory.append({"role": "tool", "tool_call_id": tid, "content": result})
            elif fname == "move_file":
                source = fargs.get("source", "")
                dest = fargs.get("dest", "")
                if not source or not dest:
                    result = "Error: 'source' and 'dest' arguments are required for move_file."
                else:
                    result = move_file_main_agent_impl(session_dir_str, source, dest)
                if verbose:
                    console.print(f"[bold cyan][Main Agent] Move file:[/bold cyan] {source} -> {dest}")
                memory.append({"role": "tool", "tool_call_id": tid, "content": result})
            else:
                memory.append({"role": "tool", "tool_call_id": tid, "content": "Unknown tool."})

    console.print("[dim]Session ended.[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    task: str = typer.Argument(..., help="Initial task for the agent (e.g. 'Build a calculator')"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model (OpenRouter format, e.g. 'openai/gpt-4o-mini')"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print all interactions (LLM calls, tool executions, etc.)"),
    debug_log_dir: str | None = typer.Option(
        None,
        "--debug-log-dir",
        help="If set, write detailed per-turn debug logs for Main and Sub Agents into this folder.",
    ),
) -> None:
    """Run the coding agent with the given task."""
    cwd = os.getcwd()
    asyncio.run(
        run_main_agent_loop(
            initial_prompt=task,
            model=model,
            cwd=cwd,
            verbose=verbose,
            debug_log_dir=debug_log_dir,
        )
    )


if __name__ == "__main__":
    app()
