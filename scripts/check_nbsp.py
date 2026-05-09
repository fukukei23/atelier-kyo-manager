# scripts/check_nbsp.py
# 先頭インデントに NBSP/TAB が残ってないか検査
import io
import os
import re
import sys

NB = "\u00a0"
PAT = re.compile(rf"^[\t{NB}]+")


def check_file(path: str) -> int:
    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            for i, line in enumerate(f, 1):
                if PAT.match(line):
                    print(f"{path}:{i}: NBSP/TAB found in leading indentation")
                    return 1
    except (FileNotFoundError, IsADirectoryError):
        return 0
    return 0


def main():
    files = [p for p in sys.argv[1:] if p.endswith(".py") and os.path.isfile(p)]
    rc = 0
    for p in files:
        rc |= check_file(p)
    sys.exit(rc)


if __name__ == "__main__":
    main()
