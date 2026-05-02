#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - 任务追踪与提醒

模型通过 TodoManager 自我追踪任务进度。当模型连续多轮忘记更新任务状态时，
nag reminder 机制会强制注入提醒消息。

执行示例：
    python agents/s03_todo_write.py

架构示意：

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

核心理念: "The agent can track its own progress -- and I can see it."
          （Agent 可以自己追踪进度——而且我能看到。）
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词：明确要求模型使用 todo 工具规划多步骤任务，
# 并要求"开始前标记为 in_progress，完成后标记为 completed"
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# -- TodoManager: 结构化的任务状态，由 LLM 通过 todo 工具写入 --
class TodoManager:
    """
    任务状态管理器。

    维护一个结构化的任务列表，支持三种状态：
    - pending:    尚未开始
    - in_progress: 正在进行（同时只能有一个）
    - completed:  已完成

    约束规则：
    - 最多 20 个任务
    - 每个任务必须有 id 和 text
    - 同一时间最多 1 个任务处于 in_progress
    - 状态不合法或 text 为空时抛出 ValueError
    """

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        """
        用模型传来的任务列表替换当前全部任务，并返回渲染后的文本。

        验证逻辑：
        1. 总数不超过 20
        2. 每个任务必须有 text
        3. 状态必须是 pending / in_progress / completed 之一
        4. in_progress 的任务最多 1 个
        """
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        """
        将任务列表渲染为人类可读的文本格式。

        格式示例：
            [ ] #1: 设计数据库模型
            [>] #2: 编写 API 接口    ← 当前正在做
            [x] #3: 初始化项目

            (1/3 completed)
        """
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# -- 工具实现函数（safe_path + 4 个文件工具，与 s02 完全一致） --
def safe_path(p: str) -> Path:
    """路径安全检查：防止模型读写工作目录之外的敏感文件。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具分发映射：s02 的 4 个工具 + 新增的 todo 工具。
# todo 不同之处：它不读写文件系统，而是修改 TodoManager 的内存状态。
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
}

# 工具定义：s02 的工具 + todo 工具。
# todo 的 items 参数是一个对象数组，每个对象包含 id、text、status 三个字段。
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
]


# -- Agent 循环：新增 nag reminder 提醒机制 --
# 与 s02 的区别：
#   1. 用 try/except 包裹 handler 调用，防止单个工具异常导致整个循环崩溃
#   2. 追踪 rounds_since_todo 计数器，如果连续 3 轮没使用 todo，注入提醒
#   3. 提醒以普通文本块的形式追加到 tool_result 中
def agent_loop(messages: list):
    """
    核心 Agent 循环（在 s02 基础上增加了 nag reminder）。

    新增逻辑：
    1. rounds_since_todo 计数器：记录连续多少轮没有使用 todo 工具
    2. 每轮检查是否使用了 todo，是则归零，否则 +1
    3. 当连续 3 轮未更新 todo 时，在工具结果末尾注入提醒消息
    """
    rounds_since_todo = 0  # 连续未使用 todo 的轮数计数
    while True:
        # 1. 调用模型（提醒在上一轮的 results 中已被注入）
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        used_todo = False
        # 2. 执行工具（带错误保护）
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    # 单个工具报错不影响其他工具的执行
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                if block.name == "todo":
                    used_todo = True  # 本轮使用了 todo，标记以便归零计数器
        # 3. Nag Reminder：连续 3 轮没用 todo 则注入提醒
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            # 以 text 类型而非 tool_result 注入——模型会在下一轮看到这条"系统消息"
            results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    交互式 REPL 入口。

    提供 s03 >> 提示符，支持 q / exit / 空输入 退出。
    与 s02 的区别：模型可以使用 todo 工具追踪进度，
    如果连续 3 轮忘记更新任务列表，会自动收到提醒。
    """
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # 打印模型最终的文本回复
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
