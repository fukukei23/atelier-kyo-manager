# scripts/fix_indent_nbsp.py
# 先頭インデントの NBSP(\u00A0) / TAB を半角スペースに正規化
# 仕様:
#  - pre-commit から渡されたファイルだけ処理 (argv)
#  - 変更があるときだけ書き込み (idempotent)
#  - 行末改行は既存を尊重、なければ 1 つ付与

import io
import os
import re
import sys

NB = "\u00a0"  # NO-BREAK SPACE
INDENT_RE = re.compile(rf"^([ \t{NB}]+)")


def fix_file(path: str) -> bool:
    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            data = f.read()
    except (FileNotFoundError, IsADirectoryError):
        return False

    orig = data
    lines = data.splitlines(keepends=True)

    fixed_lines = []
    changed = False

    for line in lines:
        m = INDENT_RE.match(line)
        if m:
            lead = m.group(1)
            new_lead = lead.replace("\t", "    ").replace(NB, " ")
            if new_lead != lead:
                line = new_lead + line[len(lead) :]
                changed = True
        fixed_lines.append(line)

    fixed = "".join(fixed_lines)

    # ファイル末尾に改行が無ければ 1 つ付与（EOF 統一用）
    if fixed and not fixed.endswith("\n"):
        fixed += "\n"
        if not orig.endswith("\n"):
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(fixed)

    return changed


def main():
    # pre-commit からのファイルリストのみ処理
    files = [p for p in sys.argv[1:] if p.endswith(".py") and os.path.isfile(p)]
    modified = []
    for p in files:
        if fix_file(p):
            modified.append(p)
            print(f"fixed-indent: {p}")
    # 変更があれば非 0 で終了 → pre-commit が「修正されたので再実行して」と案内
    sys.exit(1 if modified else 0)


if __name__ == "__main__":
    main()
