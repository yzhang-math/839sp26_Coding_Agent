"""
Sub-Agent: worker loop running in a sandbox. No globals — receives
sub_agent_memory_ref and main_agent_intervention_queue from Main Agent.
"""

import asyncio
import json
import os
from pathlib import Path

import aioconsole
from openrouter import OpenRouter
from rich.console import Console

from tools import (
    SUB_AGENT_TOOLS,
    debug_log_turn,
    execute_sub_agent_tool,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
CONFIG_FILENAMES = ("config.json",)


def _load_config() -> dict:
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
SUB_AGENT_MAX_TURNS = _config.get("sub_agent_max_turns", 30)
SUB_AGENT_SLIDING_WINDOW = _config.get("sub_agent_sliding_window", 10)

console = Console()


# ---------------------------------------------------------------------------
# LLM and prompts
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

    for m in messages:
        if m.get("role") == "system":
            system_msg = m
        elif m.get("role") == "user" and initial_user_msg is None:
            content = m.get("content", "")
            if "TASK TO COMPLETE" in content or "Task:" in content or len(content) > 100:
                initial_user_msg = m

    if system_msg:
        out.append(system_msg)
    if initial_user_msg:
        out.append(initial_user_msg)

    rest = []
    found_initial = False
    for m in messages:
        if m.get("role") == "system":
            continue
        if m.get("role") == "user" and not found_initial:
            content = m.get("content", "")
            if "TASK TO COMPLETE" in content or "Task:" in content or len(content) > 100:
                found_initial = True
                continue
        rest.append(m)

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


# ---------------------------------------------------------------------------
# Sub-Agent run
# ---------------------------------------------------------------------------

async def run_sub_agent(
    instructions: str,
    expected_deliverable: str,
    model: str,
    intervention_event: asyncio.Event,
    user_instruction_queue: list[str],
    task_name: str,
    task_dir: str,
    sub_agent_memory_ref: dict,
    main_agent_intervention_queue: asyncio.Queue,
    verbose: bool = False,
    debug_log_dir: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Run Sub-Agent in a sandbox within a task folder. Interventions go to Main Agent.
    Returns (final deliverable, Sub-Agent memory snapshot) tuple.
    """
    sandbox_dir = Path(task_dir) / f"sandbox_{task_name}"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    sandbox_dir_str = str(sandbox_dir)

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
    """Internal: run Sub-Agent logic inside a sandbox directory. Returns (deliverable, memory snapshot)."""

    def build_initial_task_message(inst: str, deliverable: str) -> str:
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

    initial_task_message = build_initial_task_message(instructions, expected_deliverable)
    memory: list[dict] = [
        {"role": "system", "content": SUB_AGENT_SYSTEM},
        {"role": "user", "content": initial_task_message},
    ]
    sub_agent_memory_ref["memory"] = memory
    sub_agent_memory_ref["update_task"] = lambda new_inst, new_deliverable: memory.__setitem__(
        1, {"role": "user", "content": build_initial_task_message(new_inst, new_deliverable)}
    )

    async def listener():
        while True:
            try:
                line = await aioconsole.ainput()
                if not line or not line.strip():
                    continue
                intervention_text = line.strip()
                console.print(f"[bold yellow]→ User intervention: {intervention_text}[/bold yellow]")
                await main_agent_intervention_queue.put(intervention_text)
                intervention_event.set()
            except (EOFError, asyncio.CancelledError):
                break

    async def sub_agent_loop() -> str:
        last_assistant_content: str = ""
        turn_count = 0
        while turn_count < SUB_AGENT_MAX_TURNS:
            if verbose:
                console.print(f"\n[bold magenta]━━━ Sub-Agent Turn {turn_count + 1} ━━━[/bold magenta]")
                console.print(f"[dim]Memory before processing: {len(memory)} messages[/dim]")

            if "task_updated" in sub_agent_memory_ref:
                sub_agent_memory_ref.pop("task_updated")
                if verbose:
                    console.print("[bold yellow][Sub-Agent] ⚠ Task instructions updated by Main Agent[/bold yellow]")
                    console.print(f"[dim]Memory before sliding window: {len(memory)} messages[/dim]")
                    if len(memory) > 1 and memory[1].get("role") == "user":
                        updated_task = memory[1].get("content", "")
                        console.print(f"[yellow]Updated task (memory[1]):[/yellow]")
                        for line in updated_task.split("\n")[:5]:
                            console.print(f"[yellow]  {line}[/yellow]")
                        if len(updated_task.split("\n")) > 5:
                            console.print(f"[yellow]  ...[/yellow]")
                memory_before = len(memory)
                memory[:] = sliding_window(memory, SUB_AGENT_SYSTEM)
                memory_after = len(memory)
                if verbose:
                    console.print(f"[dim]Memory after sliding window: {memory_before} → {memory_after} messages[/dim]")
                    has_task = any(
                        m.get("role") == "user" and "TASK TO COMPLETE" in m.get("content", "")
                        for m in memory
                    )
                    if has_task:
                        console.print(f"[green]✓ Updated task preserved in memory[/green]")
                    else:
                        console.print(f"[red]✗ WARNING: Updated task NOT found in memory after sliding window![/red]")

            if "inject_message" in sub_agent_memory_ref:
                inject_msg = sub_agent_memory_ref.pop("inject_message")
                memory.append({"role": "user", "content": inject_msg})
                if verbose:
                    console.print(f"[dim cyan][Sub-Agent] Main Agent injected message: {inject_msg[:100]}...[/dim cyan]")
                memory[:] = sliding_window(memory, SUB_AGENT_SYSTEM)

            current_messages = sliding_window(memory, SUB_AGENT_SYSTEM)

            if current_messages and current_messages[0].get("role") == "system":
                has_initial_task = any(
                    m.get("role") == "user" and "TASK TO COMPLETE" in m.get("content", "")
                    for m in current_messages
                )
                if not has_initial_task and len(memory) > 1 and memory[1].get("role") == "user":
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
                            lines = content.split("\n")
                            console.print(f"[bold yellow]  [{i}] user (TASK):[/bold yellow]")
                            for line in lines[:10]:
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

            debug_log_turn(
                debug_log_dir,
                agent="sub",
                label=f"turn_{turn_count + 1}_after_llm",
                payload={"content": content, "tool_calls": str(tool_calls)},
            )

            if not tool_calls:
                content_lower = content.lower()
                completion_keywords = ["done", "finished", "complete", "completed", "deliverable", "task complete"]
                is_complete = any(keyword in content_lower for keyword in completion_keywords)
                if is_complete and content.strip():
                    if verbose:
                        console.print("[dim cyan][Sub-Agent] Reported completion, finishing.[/dim cyan]")
                    break
                elif content.strip():
                    if verbose:
                        console.print("[dim yellow][Sub-Agent] Responded with text but no tools. Prompting to use tools...[/dim yellow]")
                    memory.append({
                        "role": "user",
                        "content": "Please use the available tools (write_file, read_file, run_shell_command, move_file, copy_file) to complete the task. Do not just describe actions - actually execute them using tools.",
                    })
                    turn_count += 1
                    continue
                else:
                    if verbose:
                        console.print("[dim cyan][Sub-Agent] Empty response, finishing.[/dim cyan]")
                    break

            if verbose:
                console.print(f"[dim cyan][Sub-Agent] Executing {len(tool_calls)} tool call(s)[/dim cyan]")

            for tc in tool_calls:
                if "task_updated" in sub_agent_memory_ref or "inject_message" in sub_agent_memory_ref:
                    if verbose:
                        console.print("[dim yellow][Sub-Agent] Main Agent updated instructions, stopping tool execution...[/dim yellow]")
                    break
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
        memory_snapshot = memory.copy()
        return deliverable, memory_snapshot
    finally:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass
