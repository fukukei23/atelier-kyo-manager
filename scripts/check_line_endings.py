#!/usr/bin/env python3
"""CRLF混入検出スクリプト（監査⑨・sentaku L1+L2 採用案 α+E）

Windows/WSL 2層運用で発生しがちな CRLF 混入を commit 前にブロックする。
既存 pre-commit パターンに倣い Python で実装（`scripts/dev/` 配下・既存と一貫）。

判定ルール:
- staged ファイルの拡張子に基づき「CRLF許容」「LF必須」を判定
- CRLF許容: .bat / .ps1 / .cmd（.gitattributes で eol=crlf 維持）
- リスト外で text 属性が auto/lf のものは LF必須
- 出力が LF必須なのに CR を含む行があれば exit 1 でブロック

設計判断:
- `pass_filenames: true`（pre-commit から staged ファイルが引数で渡る）
- ファイルが大量に呼ばれても 1ファイル1回 grep で十分（速度重視・Python 標準ライブラリのみ）
- バイナリ判定: CR を含む = 元から意図的（ZIP/PDF等）は gitattributes で filter=lfs 付与済 & text 属性なし → pre-commit の files 引数に渡らない
"""

from __future__ import annotations

import sys
from pathlib import Path

# CRLF を維持してよい拡張子（.gitattributes の eol=crlf と一致させる）
CRLF_ALLOWED_SUFFIXES: frozenset[str] = frozenset({".bat", ".ps1", ".cmd"})


def has_crlf(path: Path) -> bool:
    """ファイル内に CR を含む行があるか。

    バイナリ判定は pre-commit 側に委ねる（filter=lfs のものは files 引数に渡らない）。
    """
    try:
        with path.open("rb") as fp:
            for line in fp:
                if b"\r" in line:
                    return True
    except (OSError, UnicodeDecodeError):
        # バイナリや読めないものは pre-commit が渡さない想定・念のため無視
        return False
    return False


def main(filenames: list[str]) -> int:
    """staged ファイル群を走査し、CRLF 混入があれば exit 1 で終了。"""
    violations: list[tuple[str, str]] = []  # (ファイル名, 理由)

    for raw in filenames:
        path = Path(raw)
        suffix = path.suffix.lower()
        if suffix in CRLF_ALLOWED_SUFFIXES:
            # CRLF 維持対象 → LF 混入も検出したいので isLFonly は別ロジック必要だが
            # 現状は逆向き（CRLF混入検出）のみ。CRLF 維持対象に LF 検出は省略（誤検知コスト高）
            continue
        if has_crlf(path):
            violations.append((raw, "CRLF混入（LF必須拡張子）"))

    if violations:
        print("❌ CRLF混入を検出しました（commit ブロック）:", file=sys.stderr)
        for raw, reason in violations:
            print(f"  - {raw}: {reason}", file=sys.stderr)
        print(
            "\n解決策: `dos2unix <file>` または `sed -i 's/\\r$//' <file>` で LF 化してください。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
