# fix_nbsp.py
import pathlib, re
root = pathlib.Path(__file__).resolve().parent
nb = "\u00A0"  # NBSP
for p in root.rglob("*.py"):
    t = p.read_text(encoding="utf-8", errors="ignore")
    # 先頭インデントに含まれる NBSP とタブを 4スペースへ
    t2 = re.sub(r"(?m)^[\t"+nb+r"]+", lambda m: m.group(0).replace(nb, " ").replace("\t", "    "), t)
    if t2 != t:
        p.write_text(t2, encoding="utf-8")
        print("fixed:", p)
print("done.")
