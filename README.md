# Coding Agent (Two-Tier System)

A coding agent with a **Main Agent** (manager) and **Sub-Agent** (worker) as described in `design.md`.

## Project structure

- **`main_agent.py`** – Entry point. Orchestrates the session, creates the shared memory ref and intervention queue, and runs the Typer CLI.
- **`sub_agent.py`** – Sub-Agent worker loop (sandbox, tools, LLM). Receives `sub_agent_memory_ref` and `main_agent_intervention_queue` from the Main Agent.
- **`tools.py`** – Stateless library: file/shell helpers, tool definitions (`SUB_AGENT_TOOLS`, `get_main_agent_tools`), memo and deliverable helpers, debug logging.

## Setup

```bash
pip install -r requirements.txt
```

**API key (config file):** Copy the example config and add your key:

```bash
cp config.json.example config.json
# Edit config.json and set "api_key" to your OpenRouter API key.
```

Get your API key from [OpenRouter](https://openrouter.ai/keys).

`config.json` is read from the project directory (same folder as `main_agent.py`) or the current working directory. All parameters are optional and have defaults:

- `api_key` or `openrouter_api_key` – OpenRouter API key (used to set `OPENROUTER_API_KEY` env var if not already set)
- `model` – default model (e.g. `openai/gpt-4o-mini`); can be overridden with `--model` flag
- `sub_agent_max_turns` – maximum number of LLM turns for Sub-Agent (default: `30`)
- `sub_agent_sliding_window` – number of recent turns to keep in Sub-Agent memory (default: `10`)
- `shell_timeout` – timeout in seconds for shell commands run by Sub-Agent (default: `15`)
- `keep_sandbox` – if `true`, keep sandbox directory after execution for inspection (default: `false`)
- `sandbox_base_dir` – base directory for persistent sandboxes when `keep_sandbox` is `true` (default: current directory)

You can instead set `OPENROUTER_API_KEY` in the environment; the config file is optional. See [OpenRouter models](https://openrouter.ai/models) for available models.

## Usage

```bash
python main_agent.py main -v "Create a Python script that sorts a list of dictionaries by a specific key, and write tests for it."
```

Options:

- `--model` / `-m`: Model name (default: `openai/gpt-4o-mini`). Use any OpenRouter model string (e.g. `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, `google/gemini-pro`). See [OpenRouter models](https://openrouter.ai/models) for the full list.
- `--verbose` / `-v`: Print all interactions to console (LLM calls, tool executions, file operations, etc.). Useful for debugging.
- `--debug-log-dir`: If set, write per-turn debug logs for Main and Sub Agents into this directory.

## Behavior

1. **Main Agent** reads your task, keeps meta-memory, and can:
   - **delegate_task(instructions, expected_deliverable)** – run a Sub-Agent in a sandbox and get back a deliverable (e.g. code).
   - **write_project_file(filepath, content)** – write files into your project directory (e.g. to save code from the Sub-Agent).

2. **Sub-Agent** runs in a **temp sandbox** with:
   - `read_file` / `write_file` – sandbox-only file I/O
   - `run_shell_command` – e.g. `pytest`, `python script.py` (15s timeout)

3. **Intervention**: While the Sub-Agent is running, you can type a line and press Enter (e.g. “Make sure to handle missing keys”). That text is sent into the Sub-Agent as extra instructions and it continues with that in mind.

4. **CLI**: Rich Markdown and spinners show Main Agent thinking and Sub-Agent working. After the Main Agent replies, type your next message or `exit`/`quit`/`q` to end.

5. **Sandbox Inspection**: By default, sandboxes are temporary and auto-deleted. To inspect the sandbox:
   - Set `keep_sandbox: true` in `config.json` to preserve sandboxes after execution
   - The sandbox location is printed when Sub-Agent starts (always shown if `keep_sandbox` is true, or in verbose mode)
   - Sandboxes are created in the current directory (or `sandbox_base_dir` if specified) with names like `coding_agent_sandbox_<pid>_<timestamp>`

## Example flow

- You: `"Create a Python script that sorts a list of dicts by key, and write tests."`
- Main Agent calls `delegate_task("Write dict sorter and pytest cases", "The script code and passing pytest output")`.
- Sandbox is created; Sub-Agent writes `sorter.py`, `test_sorter.py`, runs pytest.
- (Optional) You type: `Make sure to handle missing keys` → Main Agent gets it and adjusts.
- Sub-Agent returns the code; Main Agent saves it with `write_project_file` and reports what was created.
