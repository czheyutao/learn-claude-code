#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach.
"""
s02_tool_use.py - 工具分发

相较于 s01，核心 Agent 循环完全没有变。只是在工具数组中添加了更多工具，
并引入了一个分发映射（dispatch map）来路由模型发来的工具调用。

执行示例：
    python agents/s02_tool_use.py

架构示意：

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

核心理念: "The loop didn't change at all. I just added tools."
          （循环一点没变，只是加了工具而已。）
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


def safe_path(p: str) -> Path:
    """
    路径安全检查：将用户传入的相对路径解析为工作目录下的绝对路径，
    并确保路径不会逃逸出工作目录。

    这是沙箱的安全边界——任何文件操作都必须先通过此函数校验，
    防止模型读取或写入 /etc/passwd 等系统敏感文件。
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    执行 shell 命令并返回输出。

    内置基本危险命令黑名单和 120 秒超时保护。
    输出被截断到 50000 字符，防止上下文窗口过载。
    """
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
    """
    读取文件内容并返回文本。

    limit 参数可限制返回的行数，超出的行会显示截断提示。
    读取结果限制在 50000 字符以内。
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    将 content 写入指定文件。

    如果父目录不存在则自动创建。
    写入前会通过 safe_path 进行路径安全检查。
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    精确文本替换：在文件中查找 old_text，替换为 new_text。

    只替换首次出现的位置（count=1），如果 old_text 不存在则返回错误。
    这是 s_full.py 中 Edit 工具的雏形——依赖于精确的字符串匹配而非行号。
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- 工具分发映射: {工具名称 → 处理函数} --
# 这是 s02 的核心设计：通过查表来路由工具调用，而非硬编码 if/elif。
# 新增工具只需在这里加一行映射，agent_loop 完全不用动。
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# 工具定义列表：每个工具包含名称、描述和 JSON Schema 格式的参数约束。
# 这 4 个工具构成了一个最小但完整的文件操作工具集——读、写、编辑、执行命令。
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


def agent_loop(messages: list):
    """
    核心 Agent 循环（与 s01 完全相同，没有任何修改）。

    流程：
    1. 将消息列表发送给模型
    2. 如果模型返回 tool_use，通过 TOOL_HANDLERS 查表并执行
    3. 将结果追加回消息列表
    4. 重复，直到模型不再请求工具

    与 s01 唯一的区别：工具分发从硬编码变为 TOOL_HANDLERS.get(block.name) 查表。
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 查表分发：根据工具名称找到对应的处理函数
                handler = TOOL_HANDLERS.get(block.name)
                # 如果工具不在映射表中，返回友好错误而非崩溃
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"> {block.name}:")
                print(output[:200])  # 终端只显示前 200 字符预览
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    交互式 REPL 入口。

    提供 s02 >> 提示符，支持 q / exit / 空输入 退出。
    与 s01 唯一的区别是：模型现在可以选择使用 4 种工具中的任意一种。
    """
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
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
