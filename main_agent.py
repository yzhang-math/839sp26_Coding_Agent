#!/usr/bin/env python3
"""
Main Agent: entry point and orchestration. Creates sub_agent_memory_ref and
main_agent_intervention_queue, passes them to run_sub_agent. No globals.
"""

import asyncio
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from sub_agent import openrouter_completion, run_sub_agent
from tools import (
    debug_log_turn,
    get_main_agent_tools,
    read_deliverable_impl,
    read_project_memo,
    update_persistent_memo_impl,
    write_project_file_impl,
    move_file_main_agent_impl,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_FILENAMES = ("config.json",)
_script_dir = Path(__file__).resolve().parent


def load_config() -> dict:
    """Load config; set OPENROUTER_API_KEY if not already set."""
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

app = typer.Typer(help="Coding agent: Main agent delegates tasks to a sandboxed Sub-Agent.")
console = Console()


# ---------------------------------------------------------------------------
# Main Agent system prompt
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


def main_agent_sliding_window(messages: list[dict], keep_turns: int = 10) -> list[dict]:
    """Sliding window for Main Agent: always keeps system message and initial user prompt, then last keep_turns turns."""
    if not messages:
        return []
    out = []
    system_msg = None
    initial_user_msg = None
    for m in messages:
        if m.get("role") == "system":
            system_msg = m
            break
    found_system = False
    for m in messages:
        if m.get("role") == "system":
            found_system = True
            continue
        if found_system and m.get("role") == "user" and initial_user_msg is None:
            initial_user_msg = m
            break
    if system_msg:
        out.append(system_msg)
    if initial_user_msg:
        out.append(initial_user_msg)
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
            turn = [rest[i]]
            i += 1
            while i < len(rest) and rest[i].get("role") == "tool":
                turn.append(rest[i])
                i += 1
            turns.append(turn)
        else:
            i += 1
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
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(cwd) / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)
    session_dir_str = str(session_dir)

    if verbose:
        console.print(f"[bold green]Session Workspace created:[/bold green] {session_dir_str}")

    memo_path = Path(cwd) / ".agent_memo.md"
    if not memo_path.exists():
        update_persistent_memo_impl(cwd, f"**Project Initialized.**\nInitial Goal:\n{initial_prompt}")
    else:
        update_persistent_memo_impl(cwd, f"**New Session Started.**\nWorkspace: session_{session_id}\nPrompt:\n{initial_prompt}")

    initial_memo = read_project_memo(cwd)
    initial_system = build_main_agent_system(initial_memo)
    memory: list[dict] = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_prompt},
    ]
    intervention_event = asyncio.Event()
    user_instruction_queue: list[str] = []
    sub_agent_memory_ref: dict = {}
    main_agent_intervention_queue: asyncio.Queue = asyncio.Queue()

    while True:
        current_memo = read_project_memo(cwd)
        current_system = build_main_agent_system(current_memo)
        if memory and memory[0].get("role") == "system":
            memory[0]["content"] = current_system
        else:
            memory.insert(0, {"role": "system", "content": current_system})
        memory = main_agent_sliding_window(memory, keep_turns=10)

        debug_log_turn(debug_log_dir, agent="main", label="before_llm_call", payload={"memory": memory})

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
        debug_log_turn(debug_log_dir, agent="main", label="after_llm_call", payload={"content": content, "tool_calls": str(tool_calls)})

        if not tool_calls:
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
                task_name = fargs.get("task_name", f"task_{int(time.time())}")
                context_files = fargs.get("context_files", [])
                instructions_with_conciseness = f"{instructions}\n\nIMPORTANT: Be concise. Avoid overdoing test cases and error handling."
                task_dir = Path(session_dir_str) / f"task_{task_name}"
                task_dir.mkdir(exist_ok=True)
                sandbox_dir = task_dir / f"sandbox_{task_name}"
                sandbox_dir.mkdir(parents=True, exist_ok=True)
                if context_files and isinstance(context_files, list):
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
                sub_agent_memory_ref.clear()
                sub_agent_running = True
                interventions_received_during_execution: list[str] = []

                async def process_interventions():
                    nonlocal sub_agent_running, interventions_received_during_execution
                    while sub_agent_running:
                        try:
                            intervention = await asyncio.wait_for(main_agent_intervention_queue.get(), timeout=0.1)
                            interventions_received_during_execution.append(intervention)
                            console.print(f"[bold cyan][Main Agent] Received intervention: {intervention}[/bold cyan]")
                            memory.append({
                                "role": "user",
                                "content": (
                                    f"[User intervention during Sub-Agent execution] {intervention}\n\n"
                                    f"*** SYSTEM INSTRUCTION ***\n"
                                    f"1. If this changes the task, use `update_sub_agent_task` to redirect the worker.\n"
                                    f"2. You MUST use `update_persistent_memo` to record this new requirement so it is not forgotten."
                                )
                            })
                            try:
                                if verbose:
                                    console.print(f"[bold cyan][Main Agent] Processing intervention immediately...[/bold cyan]")
                                current_memo = read_project_memo(cwd)
                                current_system = build_main_agent_system(current_memo)
                                intervention_memory = memory.copy()
                                if intervention_memory and intervention_memory[0].get("role") == "system":
                                    intervention_memory[0]["content"] = current_system
                                intervention_memory = main_agent_sliding_window(intervention_memory, keep_turns=10)
                                debug_log_turn(debug_log_dir, agent="main", label="intervention_before_llm_call", payload={"memory": intervention_memory})
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
                                            if fname == "update_sub_agent_task":
                                                new_instructions = fargs.get("new_instructions", "")
                                                new_expected_deliverable = fargs.get("new_expected_deliverable", "")
                                                if not new_instructions or not new_expected_deliverable:
                                                    result = "Error: 'new_instructions' and 'new_expected_deliverable' are required."
                                                elif "update_task" not in sub_agent_memory_ref:
                                                    result = "Error: No active Sub-Agent task to update. Call delegate_task first."
                                                else:
                                                    sub_agent_memory_ref["update_task"](new_instructions, new_expected_deliverable)
                                                    sub_agent_memory_ref["task_updated"] = True
                                                    result = f"Updated Sub-Agent task instructions. New instructions: {new_instructions[:100]}..."
                                                    if verbose:
                                                        console.print(f"[bold cyan][Main Agent] Updated Sub-Agent task:[/bold cyan] {new_instructions[:100]}...")
                                                memory.append({"role": "tool", "tool_call_id": tid, "content": result})
                                            elif fname == "update_persistent_memo":
                                                memo_content = fargs.get("content", "")
                                                if not memo_content:
                                                    result = "Error: 'content' argument is required for update_persistent_memo."
                                                else:
                                                    result = update_persistent_memo_impl(cwd, memo_content)
                                                if verbose:
                                                    console.print(f"[bold cyan][Main Agent] Updated persistent memo:[/bold cyan] {len(memo_content)} bytes")
                                                memory.append({"role": "tool", "tool_call_id": tid, "content": result})
                                    else:
                                        memory.append({"role": "assistant", "content": content or ""})
                            except Exception as e:
                                if verbose:
                                    console.print(f"[dim red][Main Agent] Error processing intervention: {e}[/dim red]")
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break

                async def run_sub_agent_task():
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

                with console.status("[bold yellow]Sub-Agent working in sandbox... (type a message and Enter to intervene)[/bold yellow]", spinner="dots"):
                    intervention_processor_task = asyncio.create_task(process_interventions())
                    try:
                        deliverable, sub_agent_memory_snapshot = await run_sub_agent_task()
                    finally:
                        sub_agent_running = False
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

                if interventions_received_during_execution:
                    console.print(f"[bold yellow]Interventions received during Sub-Agent execution: {len(interventions_received_during_execution)}[/bold yellow]")

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
                summary = (
                    f"Sub-Agent completed. Task folder: {task_dir}\nDeliverables are in {task_dir}/deliverables_{task_name}/\nSummary: {deliverable}\n\n"
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
                    if verbose:
                        console.print(f"[bold cyan][Main Agent] Updating Sub-Agent task...[/bold cyan]")
                        if "memory" in sub_agent_memory_ref and len(sub_agent_memory_ref["memory"]) > 1:
                            console.print(f"[cyan]  New instructions:[/cyan] {new_instructions[:300]}...")
                    sub_agent_memory_ref["update_task"](new_instructions, new_expected_deliverable)
                    sub_agent_memory_ref["task_updated"] = True
                    result = f"Updated Sub-Agent task instructions. New instructions: {new_instructions[:100]}..."
                memory.append({"role": "tool", "tool_call_id": tid, "content": result})

            elif fname == "read_deliverable":
                filepath = fargs.get("filepath", "")
                task_name_arg = fargs.get("task_name", "")
                if not filepath or not task_name_arg:
                    result = "Error: 'filepath' and 'task_name' arguments are required."
                else:
                    result = read_deliverable_impl(str(Path(session_dir_str) / f"task_{task_name_arg}"), task_name_arg, filepath)
                if verbose:
                    console.print(f"[bold cyan][Main Agent] Read deliverable:[/bold cyan] {filepath} ({len(result)} chars)")
                memory.append({"role": "tool", "tool_call_id": tid, "content": result})

            elif fname == "write_project_file":
                filepath = fargs.get("filepath", "")
                content = fargs.get("content", "")
                if verbose:
                    console.print(f"[bold cyan][Main Agent] Writing file:[/bold cyan] {filepath} ({len(content)} bytes)")
                result = write_project_file_impl(session_dir_str, filepath, content)
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

            elif fname == "update_persistent_memo":
                memo_content = fargs.get("content", "")
                if not memo_content:
                    result = "Error: 'content' argument is required for update_persistent_memo."
                else:
                    result = update_persistent_memo_impl(cwd, memo_content)
                if verbose:
                    console.print(f"[bold cyan][Main Agent] Updated persistent memo:[/bold cyan] {len(memo_content)} bytes")
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
    debug_log_dir: str | None = typer.Option(None, "--debug-log-dir", help="If set, write detailed per-turn debug logs for Main and Sub Agents into this folder."),
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
