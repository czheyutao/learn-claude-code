#!/usr/bin/env python3
# Harness: on-demand knowledge -- domain expertise, loaded when the model asks.
"""
s05_skill_loading.py - 技能加载（按需注入专业知识）

两层式技能注入，避免系统提示词过度膨胀：

    Layer 1（廉价）：技能名称列表放在系统提示词中（约 100 tokens/技能）
    Layer 2（按需）：完整的技能正文通过 tool_result 返回

    skills/
      pdf/
        SKILL.md          <-- frontmatter（name, description）+ 正文
      code-review/
        SKILL.md

    系统提示词：
    +--------------------------------------+
    | 你是一个编程助手。                      |
    | 可用的技能：                            |
    |   - pdf: 处理 PDF 文件...              |  <-- Layer 1: 仅元数据
    |   - code-review: 代码审查...          |
    +--------------------------------------+

    当模型调用 load_skill("pdf") 时：
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   完整的 PDF 处理指南...               |  <-- Layer 2: 完整正文
    |   步骤 1: ...                        |
    | </skill>                             |
    +--------------------------------------+

执行示例：
    python agents/s05_skill_loading.py

核心理念: "Don't put everything in the system prompt. Load on demand."
          （不要把所有的领域知识都塞进系统提示词，按需加载即可。）
"""

import os
import re
import subprocess
import yaml
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"


# -- SkillLoader: 扫描 skills/<name>/SKILL.md 文件，解析 YAML frontmatter --
class SkillLoader:
    """
    技能加载器。

    在初始化时递归扫描 skills 目录下的所有 SKILL.md 文件，
    解析 YAML frontmatter 获取技能名称（name）、描述（description）、标签（tags），
    并将正文部分缓存起来供按需加载。

    两层设计：
        get_descriptions() → Layer 1：放入系统提示词，仅含名称和描述
        get_content(name)  → Layer 2：通过 load_skill 工具按需返回完整正文
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self._load_all()

    def _load_all(self):
        """
        递归扫描 skills 目录下的所有 SKILL.md 文件并解析。
        使用目录名作为默认的技能标识符。
        """
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        """
        解析 YAML frontmatter。

        SKILL.md 文件以 --- 分隔符包裹 YAML 元数据，格式如下：
            ---
            name: pdf
            description: 处理 PDF 文件的指南
            tags: document, pdf
            ---
            技能正文从这里开始...
        """
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """Layer 1：生成技能描述列表，嵌入到系统提示词中（廉价）。"""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2：根据技能名称返回完整的技能正文（按需、昂贵）。"""
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(SKILLS_DIR)

# Layer 1：将技能描述信息注入系统提示词
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


# -- 工具实现函数（与 s04 一致，新增 load_skill） --
def safe_path(p: str) -> Path:
    """路径安全检查：将相对路径解析为工作目录下的绝对路径，防止路径逃逸。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令并返回输出，含基本的危险命令黑名单和 120 秒超时保护。"""
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
    """读取文件内容，支持 limit 限制行数，结果截断至 50000 字符。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """将内容写入指定文件，父目录不存在时自动创建。"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在文件中精确替换文本（仅替换首次出现的位置）。"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- 工具分发映射 --
# 与 s04 的区别：去掉了 task 工具（子代理），新增了 load_skill 工具（技能加载）。
# 这意味着 s05 的 Agent 专注于"自己完成任务 + 按需加载领域知识"，
# 不再具备创建子代理的能力。
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}},
]


# -- Agent 循环 --
# 与 s04 相比没有任何结构变化。唯一的区别是 TOOLS 中包含 load_skill，
# 因此模型可以在需要领域知识时通过查表调用 skill 加载器。
# 核心循环依然是：调用模型 → 执行工具 → 追加结果 → 重复。
def agent_loop(messages: list):
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
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    交互式 REPL 入口。

    提供 s05 >> 提示符，支持 q / exit / 空输入 退出。
    与 s04 的区别：模型现在可以加载技能知识来指导自己的工作。
    建议尝试以下输入：
        "What skills are available?"
        "Load the agent-builder skill"
        "I need to do a code review"
    """
    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
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
