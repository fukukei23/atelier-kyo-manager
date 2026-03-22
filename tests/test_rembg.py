# test_rembg.py  ── そのまま保存して実行してください ───────────────
# rembg を使い 1 枚だけ背景除去 → REMBG_*.png を同じフォルダに書き出します
# ---------------------------------------------------------------
# 事前準備:  pip install rembg pillow

import pytest
import sys
from pathlib import Path
from PIL import Image

# ★★ テストしたい JPEG フルパスをここに貼り付ける ★★
# Windows環境でのみ実行
TARGET = Path(r"D:\catalog_images\GUCCI\80910\catalog_80910\93b42ae4-302b-4efb-bd8e-712e7c78b6e5_20250516022114122640.jpg")

def test_rembg_background_removal():
    """rembg を使用した背景除去テスト（Windows環境のみ）"""
    if sys.platform != "win32":
        pytest.skip("rembg test is only available on Windows")

    if not TARGET.exists():
        pytest.skip(f"テストデータファイルが見つかりません: {TARGET}")

    try:
        from rembg import remove
    except ImportError:
        pytest.skip("rembg not installed")

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

    # 出力ファイルが作成されたことを確認
    assert OUT.exists(), f"出力ファイルが作成されませんでした: {OUT}"
    print("[OK] rembg 背景除去テスト成功！")
