# test_rembg.py  ── そのまま保存して実行してください ───────────────
# rembg を使い 1 枚だけ背景除去 → REMBG_*.png を同じフォルダに書き出します
# ---------------------------------------------------------------
# 事前準備:  pip install rembg pillow

from pathlib import Path
from PIL import Image
from rembg import remove

# ★★ テストしたい JPEG フルパスをここに貼り付ける ★★
TARGET = Path(r"D:\catalog_images\GUCCI\80910\catalog_80910\93b42ae4-302b-4efb-bd8e-712e7c78b6e5_20250516022114122640.jpg")

# ---------------------------------------------------------------
if not TARGET.exists():
    raise FileNotFoundError(f"ファイルが見つかりません:\n{TARGET}")

print(f"[INFO] 入力: {TARGET}")
print(f"[INFO] サイズ: {TARGET.stat().st_size // 1024} KB")

# 背景除去
img_in   = Image.open(TARGET).convert("RGBA")   # CMYK 画像対策で convert
img_out  = remove(img_in)

# 出力ファイル名 例: REMBG_93b42ae4-....png
OUT = TARGET.parent / f"REMBG_{TARGET.stem}.png"
img_out.save(OUT)

print(f"[OK] 出力: {OUT}")
print(f"[OK] サイズ: {OUT.stat().st_size // 1024} KB")
