#!/usr/bin/env python3
"""
s01_agent_loop.py - Agent 核心循环（最小化实现）

AI 编程助手的核心模式——"思考-行动-观察"循环：

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        执行工具调用
        将结果追加回消息列表

该文件展示了 Agent 的最简形式：没有计划、没有子代理、没有记忆，
只有一个 while 循环来驱动模型与工具的交互。

执行示例：
    python agents/s01_agent_loop.py
"""

import os
import subprocess

# 设置 readline 使交互式输入更友好（特别是 macOS libedit 的退格键问题）
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(override=True)

# 如果使用了自定义 API 地址（如 DeepSeek），则清除 auth token，
# 因为自定义端点不使用 Anthropic 的认证方式
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 初始化 Anthropic 客户端（可指定自定义 base_url）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 从环境变量读取模型 ID（如 claude-sonnet-4-6 或 deepseek-v4-flash）
MODEL = os.environ["MODEL_ID"]

# 系统提示词：告诉模型它的身份和行为方式
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# 定义工具：模型可以通过 tool_use 来请求执行 bash 命令
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def run_bash(command: str) -> str:
    """
    执行 bash 命令并返回输出结果。

    包含基本的危险命令检查和超时保护。
    输出被截断到 50000 字符，防止上下文窗口过载。
    """
    # 列出一组明显危险的操作，直接拦截
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,         # 防止命令卡死
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def agent_loop(messages: list):
    """
    核心 Agent 循环。

    这是整个 AI Agent 的最核心模式：
    1. 将消息列表发送给模型，获取响应
    2. 如果模型返回了 tool_use，执行对应的工具
    3. 将工具结果追加回消息列表
    4. 重复，直到模型主动停止（stop_reason != "tool_use"）

    参数:
        messages: 完整的对话消息列表，会在此列表上原地追加内容
    """
    while True:
        # 1. 调用模型
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # 2. 将模型的回复追加到消息列表中
        messages.append({"role": "assistant", "content": response.content})
        # 3. 如果模型不再需要调用工具，则本轮对话结束
        if response.stop_reason != "tool_use":
            return
        # 4. 遍历所有 tool_use 块，依次执行
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                # 终端只显示输出前 200 字符，但送回的完整结果会包含所有内容
                print(output[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        # 5. 将工具结果以 user 角色送还给模型，进入下一轮
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    交互式 REPL 入口。

    提供一个 s01 >> 提示符，接收用户输入后启动 agent_loop。
    支持 q / exit / 空输入 退出。
    """
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
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
