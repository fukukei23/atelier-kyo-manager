#!/usr/bin/env python3
"""
atelier-kyo-manager デモ録画スクリプト v2
==========================================
cast → Pillow → ffmpeg で CLI デモ GIF/MP4 を生成

使用方法:
    python scripts/record_demo.py              # 全シーン生成
    python scripts/record_demo.py --scene 1     # シーン1のみ
    python scripts/record_demo.py --list        # シーン一覧
"""

import os
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================================
# 設定
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEMO_DIR = PROJECT_ROOT / "docs" / "demo"
CAST_DIR = DEMO_DIR / "casts"
PNG_DIR = DEMO_DIR / "frames"
GIF_DIR = DEMO_DIR / "gifs"
MP4_DIR = DEMO_DIR / "mp4"

# Pillow設定
FPS = 10
WIDTH = 80
HEIGHT = 24

# テーマ（Solarized Dark）
BG_COLOR = (40, 44, 52)
TEXT_COLOR = (220, 223, 228)
PROMPT_COLOR = (86, 156, 214)
HEADER_COLOR = (106, 153, 78)
GREEN = (106, 153, 78)
RED = (204, 85, 85)

# ============================================================================
# シーン定義
# ============================================================================
SCENES = [
    {
        "id": 1,
        "name": "dashboard",
        "title": "ダッシュボード概要",
        "cast_file": "01_dashboard.cast",
        "gif_file": "01_dashboard.gif",
        "mp4_file": "01_dashboard.mp4",
        "duration": 20,
        "output": """=== Atelier KYO Manager ===

$ python -m app.cli dashboard

📊 売上ダッシュボード
---------------------------
本日: ¥45,200 (目標 ¥50,000)
今月: ¥1,234,500 (前月比 +12.3%)
未処理注文: 3件

📦 商品一覧
---------------------------
SKU-001  coach bag  在庫:12  ¥89,000
SKU-002  wallet    在庫:8   ¥32,000
SKU-003  sneakers  在庫:5   ¥67,000

✅ システム正常稼働"""
    },
    {
        "id": 2,
        "name": "orders",
        "title": "注文ステートマシン",
        "cast_file": "02_orders.cast",
        "gif_file": "02_orders.gif",
        "mp4_file": "02_orders.mp4",
        "duration": 25,
        "output": """=== 注文管理（18日ルール） ===

$ python -m app.cli orders --list

📋 注文一覧
---------------------------
#1023  pending  → 2026-06-12期限
#1024  sourcing → 2026-06-15期限
#1025  shipped  → 2026-06-20到着予定

🔄 自動状态遷移
---------------------------
pending → sourcing → shipped → completed
         ↕         ↕
      Buyandship 出庫待ち  通関中

⚠️ 18日ルール適用:
  カード払い: 18日以内に出庫
  銀行振込み: 18日+N日以内"""
    },
    {
        "id": 3,
        "name": "scraping",
        "title": "価格スクレイピング",
        "cast_file": "03_scraping.cast",
        "gif_file": "03_scraping.gif",
        "mp4_file": "03_scraping.mp4",
        "duration": 20,
        "output": """=== リサーチ（価格取得） ===

$ python -m app.cli research --country US

🌐 スクレイピング実行中...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
URL: https://example.com/bags
✅ 取得成功: 12商品

📊 結果（USD → JPY 149.5）
---------------------------
Coach Tabby    $298 → ¥44,551
Coach Swagger  $245 → ¥36,627
Michael Kors    $189 → ¥28,240

💾 データベース保存完了
   price_cache: 12件更新"""
    },
    {
        "id": 4,
        "name": "tests",
        "title": "テストスイート",
        "cast_file": "04_tests.cast",
        "gif_file": "04_tests.gif",
        "mp4_file": "04_tests.mp4",
        "duration": 25,
        "output": """=== テスト実行 ===

$ python -m pytest tests/ -x -q

tests collected: 2070
passed        2068
failed           0

✅ 全テスト通過 (99.9%)
📊 カバレッジ: 78.5%"""
    },
]

# ============================================================================
# ユーティリティ
# ============================================================================

def ensure_dirs():
    for d in [CAST_DIR, PNG_DIR, GIF_DIR, MP4_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"📁 出力ディレクトリ: {DEMO_DIR}")


def cast_to_png_frames(cast_path: Path, png_dir: Path, fps: int, duration: float) -> list:
    """cast → PNGフレーム（Pillow高速描画）"""
    print(f"   🖼️  PNGフレーム生成中...")
    png_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw, ImageFont

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font_size = 14
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    line_h = font_size + 6
    char_w = font_size // 2 + 2
    margin = 20
    canvas_w = WIDTH * char_w + margin * 2
    canvas_h = HEIGHT * line_h + margin * 2

    events = []
    if cast_path.exists():
        with open(cast_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if isinstance(event, list) and len(event) >= 3:
                        events.append(event)
                except:
                    continue

    total_frames = int(duration * fps)
    frame_times = [i / fps for i in range(total_frames)]

    png_files = []
    for i, t in enumerate(frame_times):
        frame_path = png_dir / f"frame_{i:04d}.png"
        img = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
        draw = ImageDraw.Draw(img)

        text_y = margin
        for event in events:
            event_time, event_type, text = event[0], event[1], event[2]
            if event_time > t:
                break
            if event_type == "o" and text:
                for line_text in text.replace("\r\n", "\n").replace("\r", "").split("\n"):
                    if line_text.strip():
                        if line_text.startswith("$"):
                            color = PROMPT_COLOR
                        elif line_text.startswith("===") or line_text.startswith("---"):
                            color = HEADER_COLOR
                        elif line_text.startswith("✅") or line_text.startswith("passed"):
                            color = GREEN
                        elif line_text.startswith("⚠️") or line_text.startswith("❌"):
                            color = RED
                        else:
                            color = TEXT_COLOR
                        draw.text((margin, text_y), line_text, font=font, fill=color)
                        text_y += line_h
                        if text_y > canvas_h - margin:
                            break

        img.save(frame_path, "PNG")
        png_files.append(frame_path)

        if (i + 1) % 50 == 0:
            print(f"   📊 {i + 1}/{total_frames} フレーム")

    print(f"   ✅ PNG生成完了: {len(png_files)} フレーム")
    return png_files


def png_to_gif(png_files: list, gif_path: Path, fps: int) -> bool:
    if not png_files:
        return False
    print(f"   🎬 GIF生成中...")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        for p in png_files:
            f.write(f"file '{p}'\n")
        concat_file = f.name

    ffmpeg = os.path.expanduser("~/.local/bin/ffmpeg")
    try:
        subprocess.run([
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-vf", f"fps={fps},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0", str(gif_path)
        ], check=True, capture_output=True)
        print(f"   ✅ GIF生成: {gif_path.name} ({gif_path.stat().st_size // 1024}KB)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ GIF生成失敗")
        return False
    finally:
        Path(concat_file).unlink()


def png_to_mp4(png_files: list, mp4_path: Path, fps: int) -> bool:
    if not png_files:
        return False
    print(f"   🎬 MP4生成中...")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        for p in png_files:
            f.write(f"file '{p}'\n")
        concat_file = f.name

    ffmpeg = os.path.expanduser("~/.local/bin/ffmpeg")
    try:
        subprocess.run([
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-vf", f"fps={fps}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            str(mp4_path)
        ], check=True, capture_output=True)
        print(f"   ✅ MP4生成: {mp4_path.name} ({mp4_path.stat().st_size // 1024}KB)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ MP4生成失敗")
        return False
    finally:
        Path(concat_file).unlink()


def generate_sample_cast(scene: dict) -> Path:
    """asciinema v2形式cast生成"""
    cast_path = CAST_DIR / scene["cast_file"]

    with open(cast_path, "w") as f:
        header = {"version": 2, "width": WIDTH, "height": HEIGHT,
                  "timestamp": int(time.time()), "env": {"TERM": "xterm-256color"}}
        f.write(json.dumps(header) + "\n")

        timestamp = 0.0
        for line in scene["output"].split("\n"):
            f.write(json.dumps([round(timestamp, 4), "o", line + "\r\n"]) + "\n")
            timestamp += 0.15

        f.write(json.dumps([round(timestamp, 4), "o", "$ "]) + "\n")

    print(f"   ✅ Cast生成: {cast_path.name}")
    return cast_path


# ============================================================================
# メイン処理
# ============================================================================

def process_scene(scene: dict, generate: bool = True) -> dict:
    result = {"scene": scene["name"], "success": False}
    print(f"\n🎬 シーン {scene['id']}: {scene['title']}")

    cast_path = CAST_DIR / scene["cast_file"]
    if generate or not cast_path.exists():
        cast_path = generate_sample_cast(scene)

    png_dir = PNG_DIR / scene["name"]
    png_files = cast_to_png_frames(cast_path, png_dir, FPS, scene["duration"])
    if not png_files:
        return result

    gif_path = GIF_DIR / scene["gif_file"]
    mp4_path = MP4_DIR / scene["mp4_file"]

    if png_to_gif(png_files, gif_path, FPS):
        result["gif"] = str(gif_path)
    if png_to_mp4(png_files, mp4_path, FPS):
        result["mp4"] = str(mp4_path)

    result["success"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description="atelier Demo Recorder")
    parser.add_argument("--scene", type=int, choices=range(1, 5),
                        help="特定シーンのみ処理")
    parser.add_argument("--list", action="store_true", help="シーン一覧")

    args = parser.parse_args()
    ensure_dirs()

    if args.list:
        print("\n📋 シーン一覧:")
        for s in SCENES:
            print(f"  {s['id']}. {s['title']} ({s['duration']}秒)")
        return

    scenes = [s for s in SCENES if not args.scene or s["id"] == args.scene]
    print(f"\n🎬 atelier Demo Recorder — 対象: {[s['id'] for s in scenes]}")

    results = []
    for scene in scenes:
        try:
            result = process_scene(scene, generate=True)
            results.append(result)
        except Exception as e:
            print(f"\n❌ エラー: {e}")
            continue

    success = sum(1 for r in results if r["success"])
    print(f"\n✅ 成功: {success}/{len(results)} シーン")


if __name__ == "__main__":
    main()