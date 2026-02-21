System Design & Architecture: CLI Coding Agent
1. Overview

This project implements a highly modular, framework-free command-line coding agent. Rather than relying on heavy abstraction layers like LangChain or CrewAI, the system is built entirely on native Python asyncio and raw LLM API calls.

The architecture employs a Hierarchical Two-Tier Agent System consisting of a Main Agent (The Manager) and temporary Sub-Agents (The Workers). This design elegantly handles token context limits, isolates code execution safely, and allows for real-time human-in-the-loop steering.
2. Core Architecture: The Two-Tier System
The Main Agent (Orchestrator)

    Role: Interacts with the user, maintains the long-term project state, and delegates discrete tasks.

    Capabilities: It has access to file management tools (move_file, write_project_file, read_deliverable) and orchestration tools (delegate_task, update_sub_agent_task, update_persistent_memo).

    Execution: Runs the primary event loop and manages the high-level session_workspace.

The Sub-Agent (Sandboxed Worker)

    Role: Executes specific coding, testing, or debugging tasks assigned by the Main Agent.

    Capabilities: Operates strictly within a temporary jail. It can read_file, write_file, and run_shell_command only within its designated sandbox.

    Lifecycle: Ephemeral. It spins up to complete a task, hands the files to a deliverables folder, summarizes its work, and is then destroyed (preventing context bloat).

3. The 3-Tier Memory Architecture

To prevent API crashes (TokenLimitExceeded) and hallucination, memory is managed across three distinct tiers:

    Working Memory (Tier 1 - The Sliding Window):
    Both agents use a custom sliding_window() algorithm for their API payload. The algorithm permanently anchors the System Prompt and the Initial Task at the front of the array, but dynamically drops the "middle" of the conversation, keeping only the last ~10 back-and-forth turns.

    Environmental Memory (Tier 2 - The File System):
    Agents are taught to treat the hard drive as their primary state. If a Sub-Agent forgets a variable name, it must use the read_file tool to check the disk rather than relying on conversation history.

    Persistent Memory (Tier 3 - The Project Memo):
    The Main Agent maintains an .agent_memo.md file in the project root. On every turn, the contents of this file are dynamically injected into the Main Agent's System Prompt. The Main Agent uses the update_persistent_memo tool to record completed milestones and user interventions, allowing it to "remember" the project state even if the CLI is closed and restarted days later.

4. Workspace & File Management

The system enforces a strict, hierarchical directory structure to prevent destructive path traversal and maintain project hygiene.

"""
my_project/
├── .agent_memo.md                 # Persistent long-term memory
└── session_20260221_103000/       # Isolated session workspace
    ├── main.py                    # Final aggregated code
    ├── task_generator/            # Sub-Agent task container
    │   ├── sandbox_generator/     # Jailed execution environment
    │   └── deliverables_generator/# Handoff directory
    └── task_analyzer/
"""

Sandboxing: When delegate_task is called, a sandbox is created. Tools enforce a jail using pathlib relative-path resolution. The run_shell_command tool enforces cwd=sandbox_dir so executed code (pytest, python) cannot easily escape.

    Context Passing (context_files): If a Sub-Agent needs to modify or test existing code, the Main Agent explicitly passes an array of context_files. The system securely copies these files from the session root down into the Sub-Agent's sandbox before it wakes up.

    The "Fail Fast" Protocol: Sub-Agents are strictly prompted to immediately halt and report failure if they are missing context files, prompting the Main Agent to re-delegate with the proper files rather than hallucinating code.

5. Asynchronous Real-Time Intervention

A standout feature is the Human-in-the-Loop Async Intervention system, allowing the user to steer the agent while it is actively generating code.

    The Listener: While the Sub-Agent is locked in its ReAct loop, a concurrent asyncio task listens to the terminal (aioconsole.ainput).

    The Intercept: When the user types a command (e.g., "Wait, use merge sort instead"), it is placed in a main_agent_intervention_queue.

    The Pivot: The Main Agent immediately wakes up, processes the intervention, logs it to the persistent memo, and executes update_sub_agent_task.

    Shared State: Because state is managed via Dependency Injection (passing sub_agent_memory_ref dicts instead of using global variables), the Main Agent instantly rewrites the Sub-Agent's foundational prompt mid-flight. The Sub-Agent seamlessly pivots its approach on the very next LLM call.

6. Codebase Organization (Modularity)

The codebase is split into three stateless, tightly scoped files:

    tools.py: A pure library of tool implementations (file I/O, shell execution, memory reading) and their JSON schema definitions. It contains zero LLM logic and no global state. Includes a custom SafeEncoder to prevent json.dump crashes during debug logging.

    sub_agent.py: Contains the sandboxed worker loop, the Sub-Agent's sliding window logic, and its specific system prompts.

    main_agent.py: The CLI entry point. Initializes the state objects (asyncio.Queue, session directories), houses the Typer CLI commands, and runs the overarching orchestrator loop.

This separation of concerns eliminates race conditions and negates the need for global variables entirely.
