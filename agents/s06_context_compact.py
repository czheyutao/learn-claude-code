#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - 上下文压缩（让 Agent 永不忘记的秘诀）

三种压缩策略，让 Agent 可以在有限的上下文窗口中无限工作：

    每轮对话后：
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [第1层: micro_compact]        (静默执行，每轮)
      将超过最后 3 轮的 tool_result 内容
      替换为 "[Previous: used {tool_name}]"
            |
            v
    [检查: token > 50000?]
       |               |
       否              是
       |               |
       v               v
    继续         [第2层: auto_compact]
                  保存完整对话到 .transcripts/
                  请 LLM 总结对话内容
                  用摘要替换所有消息
                        |
                        v
                [第3层: compact 工具]
                  模型主动调用 compact 工具 -> 立即压缩
                  与 auto_compact 逻辑相同，但由模型手动触发

                  示意图：
                  ┌──────┐  token > 50K  ┌──────────┐
                  │ 对话  │ ─────────────→ │ 保存原文  │
                  │ 历史  │               │ 到磁盘    │
                  └──┬───┘               └─────┬────┘
                     │                        │
                     │ 每轮（micro_compact）    │
                     ▼                        ▼
                  ┌──────────┐           ┌──────────┐
                  │ 替换旧    │           │ LLM 总结  │
                  │ tool_    │           │ 对话内容  │
                  │ result   │           └─────┬────┘
                  └──────────┘               │
                                             ▼
                                          ┌──────────┐
                                          │ 摘要替换  │
                                          │ 原始消息  │
                                          └──────────┘

核心理念: "The agent can forget strategically and keep working forever."
          （Agent 可以策略性地"遗忘"——从而永远工作下去。）
"""

import json
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

# -- 压缩阈值配置 --
THRESHOLD = 50000          # token 超过此值触发 auto_compact
TRANSCRIPT_DIR = WORKDIR / ".transcripts"  # 原始对话存档目录
KEEP_RECENT = 3            # micro_compact 保留最近 N 轮 tool_result
PRESERVE_RESULT_TOOLS = {"read_file"}  # 这些工具的结果不压缩（保留引用内容）


def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数：约每 4 字符 = 1 token。"""
    return len(str(messages)) // 4


# -- 第1层: micro_compact - 将旧 tool_result 替换为占位符 --
def micro_compact(messages: list) -> list:
    """
    每轮静默执行。遍历所有 tool_result，将超过 KEEP_RECENT 轮的
    旧结果替换为简短占位符，以减少 token 消耗。
    但 read_file 的结果会被保留——压缩它会导致 Agent 重新读取文件。
    """
    # 收集所有 tool_result 的位置
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    # 如果结果数量不足保留阈值，不压缩
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # 通过 tool_use_id 反向查找对应的工具名称
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    # 只处理较旧的结果（保留最近 KEEP_RECENT 轮）
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        # 跳过短结果（<100 字符的无需压缩）
        if not isinstance(result.get("content"), str) or len(result["content"]) <= 100:
            continue
        tool_id = result.get("tool_use_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")
        # 保留白名单工具（如 read_file）的完整结果
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue
        result["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- 第2层: auto_compact - 保存对话、总结、替换 --
def auto_compact(messages: list) -> list:
    """
    当 token 超过阈值时触发：
    1. 将完整对话保存到 .transcripts/ 目录（时间戳命名）
    2. 让 LLM 总结当前对话的进展、状态和关键决策
    3. 用一条包含摘要的 user 消息替换全部对话历史
    """
    # 保存完整对话到磁盘，便于后续回溯
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # 提取最后 80K 字符交给 LLM 总结（太长会导致请求超限）
    conversation_text = json.dumps(messages, default=str)[-80000:]
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    if not summary:
        summary = "No summary generated."
    # 用摘要替换所有历史消息
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]


# -- 工具实现 --
def safe_path(p: str) -> Path:
    """安全检查：确保路径不逃逸工作目录。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """执行 shell 命令（带危险命令过滤）。"""
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
    """读取文件内容，可选限制行数。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    """写入文件（自动创建父目录）。"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在文件中替换指定文本。"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- 工具注册表：名称 -> 处理函数 --
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",
}

# -- 工具定义（含 compact 工具，供模型手动触发压缩） --
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


def agent_loop(messages: list):
    """
    Agent 主循环，在标准循环基础上注入三层压缩机制：

    第1层：micro_compact — 每轮 LLM 调用前静默执行，压缩旧 tool_result
    第2层：auto_compact — 当 token 估算超过阈值时，自动总结并替换历史
    第3层：manual compact — 模型调用 compact 工具触发的主动压缩
    """
    while True:
        # 第1层：每轮调用 LLM 前执行微压缩
        micro_compact(messages)
        # 第2层：如果 token 超过阈值，触发自动压缩
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        # 如果模型没有调用工具，本轮结束
        if response.stop_reason != "tool_use":
            return
        # 执行所有工具调用
        results = []
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    # 第3层：手动压缩标记（工具先返回，循环外执行压缩）
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
        # 第3层（续）：如果模型调用了 compact，执行手动压缩后结束本轮
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


if __name__ == "__main__":
    """交互式 REPL：用户输入问题，Agent 循环处理，输出最终回复。"""
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
