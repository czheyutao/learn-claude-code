#!/usr/bin/env python3
"""
最长公共子序列 (Longest Common Subsequence, LCS)

实现功能：
  1. 计算两个字符串的最长公共子序列长度
  2. 还原出具体的最长公共子序列
  3. 支持可视化 DP 表格

动态规划思路：
  dp[i][j] = 在 text1[:i] 和 text2[:j] 中的 LCS 长度

  状态转移方程：
    if text1[i-1] == text2[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
    else:
        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

示例：
    text1 = "abcde"
    text2 = "ace"
    LCS = "ace", 长度为 3
"""


def lcs_length(text1: str, text2: str) -> int:
    """
    计算最长公共子序列的长度。
    使用一维数组优化空间复杂度至 O(n)。
    """
    m, n = len(text1), len(text2)
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0  # 相当于 dp[i-1][j-1]
        for j in range(1, n + 1):
            temp = dp[j]
            if text1[i - 1] == text2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def lcs_table(text1: str, text2: str) -> tuple[list[list[int]], list[list[str]]]:
    """
    构建完整的 DP 表（长度表）和方向表（用于路径回溯）。
    返回 (dp 表, 方向表)。

    方向标记：
        '↖' — 字符匹配，对角线方向
        '←' — 来自左方
        '↑' — 来自上方
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    direction = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                direction[i][j] = "↖"
            elif dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                direction[i][j] = "↑"
            else:
                dp[i][j] = dp[i][j - 1]
                direction[i][j] = "←"

    return dp, direction


def lcs_reconstruct(text1: str, text2: str) -> str:
    """
    还原最长公共子序列的具体内容。
    """
    dp, direction = lcs_table(text1, text2)
    i, j = len(text1), len(text2)
    chars = []

    while i > 0 and j > 0:
        if direction[i][j] == "↖":
            chars.append(text1[i - 1])
            i -= 1
            j -= 1
        elif direction[i][j] == "↑":
            i -= 1
        else:  # "←"
            j -= 1

    return "".join(reversed(chars))


def lcs_all(text1: str, text2: str) -> dict:
    """
    一次性返回所有 LCS 相关信息。
    """
    length = lcs_length(text1, text2)
    sequence = lcs_reconstruct(text1, text2)
    return {
        "text1": text1,
        "text2": text2,
        "length": length,
        "lcs": sequence,
    }


def print_dp_table(text1: str, text2: str) -> None:
    """
    打印 DP 表（含行列标签），方便学习理解。
    """
    dp, direction = lcs_table(text1, text2)
    m, n = len(text1), len(text2)

    print("\n==================== DP 表 ====================")
    print(f"  text1 = '{text1}'  (行, i)")
    print(f"  text2 = '{text2}'  (列, j)")
    print()

    # 表头
    header = [" ", "∅"] + list(text2)
    print(f"{'':>4}", end="")
    for h in header:
        print(f"{h:>4}", end="")
    print()

    # 分隔线
    print(f"{'':>4}" + "─" * (4 * (n + 2)))

    # 行标签 + 数据
    row_labels = ["∅"] + list(text1)
    for i in range(m + 1):
        print(f"{row_labels[i]:>4}", end="")
        for j in range(n + 1):
            val = dp[i][j]
            dir_char = direction[i][j] if direction[i][j] else "·"
            print(f"{dir_char}{val:>3}", end="")
        print()

    print(f"\n最长公共子序列长度: {dp[m][n]}")
    print(f"最长公共子序列: '{lcs_reconstruct(text1, text2)}'")
    print("================================================\n")


def run_tests() -> None:
    """运行内置测试用例"""
    test_cases = [
        ("abcde", "ace", 3, "ace"),
        ("abc", "abc", 3, "abc"),
        ("abc", "def", 0, ""),
        ("abcdef", "acf", 3, "acf"),
        ("AGGTAB", "GXTXAYB", 4, "GTAB"),
        ("", "abc", 0, ""),
        ("abc", "", 0, ""),
        ("aaaa", "aa", 2, "aa"),
    ]

    all_pass = True
    for t1, t2, exp_len, exp_seq in test_cases:
        result = lcs_all(t1, t2)
        ok = result["length"] == exp_len and result["lcs"] == exp_seq
        status = "✅" if ok else "❌"
        if not ok:
            all_pass = False
        print(f"  {status} LCS('{t1}', '{t2}') = "
              f"len={result['length']}, seq='{result['lcs']}' "
              f"(expected len={exp_len}, seq='{exp_seq}')")

    print(f"\n{'全部通过!' if all_pass else '有测试失败!'}")


# ─────────────────────────────────────────────────────────
# 交互式演示
# ─────────────────────────────────────────────────────────
def interactive_mode() -> None:
    """
    交互模式：用户输入两个字符串，程序输出 LCS 信息和 DP 表。
    """
    print("\n" + "=" * 60)
    print("  最长公共子序列 (LCS) - 交互式演示")
    print("=" * 60)
    print("输入 q 或 exit 退出\n")

    while True:
        t1 = input("text1: ").strip()
        if t1.lower() in ("q", "exit", ""):
            break
        t2 = input("text2: ").strip()
        if t2.lower() in ("q", "exit", ""):
            break

        result = lcs_all(t1, t2)
        print(f"\n📌 结果: LCS = '{result['lcs']}', 长度 = {result['length']}")
        print_dp_table(t1, t2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        # 命令行模式: python lcs.py text1 text2
        t1, t2 = sys.argv[1], sys.argv[2]
        result = lcs_all(t1, t2)
        print(f"LCS('{t1}', '{t2}') = '{result['lcs']}'  (长度={result['length']})")
    elif len(sys.argv) == 2 and sys.argv[1] in ("-t", "--test"):
        # 测试模式
        run_tests()
    else:
        # 交互模式
        interactive_mode()
