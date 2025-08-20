import asyncio, logging, re, sys
from pathlib import Path
import yaml
from playwright.async_api import async_playwright

# ────────── 設定 ──────────
HEADLESS = True
TIMEOUT  = 60_000
ROOT     = Path(__file__).resolve().parents[1]
CONF     = ROOT / "configs"
STORAGE  = CONF / "storage.json"
OUT      = CONF / "warehouses.yaml"
CONF.mkdir(exist_ok=True)

URL_LIST = "https://www.buyandship.co.jp/account/v2020/warehouse/"
STOPPED  = {"UK", "CA"}         # 転送停止倉庫（必要なら編集）

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

# ────────── レスポンスフィルタ ──────────
def is_warehouse_api(resp):
    return ("warehouse" in resp.url) and resp.headers.get("content-type","").startswith("application/json")

# ────────── メイン ──────────
async def main():
    if not STORAGE.exists():
        sys.exit("storage.json がありません。先に codegen で作成してください")

    async with async_playwright() as p:
        br  = await p.chromium.launch(headless=HEADLESS)
        ctx = await br.new_context(storage_state=str(STORAGE))
        page = await ctx.new_page()

        # ページを開き、倉庫一覧 API をキャッチ
        logging.info("倉庫一覧ページをロード中…")
        fut = page.expect_response(is_warehouse_api, timeout=TIMEOUT)
        await page.goto(URL_LIST, wait_until="domcontentloaded", timeout=TIMEOUT)

        try:
            async with fut as resp:                    # API レスポンスを取得
        data = await resp.json()
        except asyncio.TimeoutError:
            sys.exit("❌ 住所 API を検出できませんでした。storage.json が失効している可能性")
        await br.close()

    # レスポンス構造が配列 or dict どちらでも吸収
    if isinstance(data, list):
        rows = {row.get("countryCode",""): row for row in data}
    else:
        rows = data

    result = {}
    for code, row in rows.items():
        if code in STOPPED:
            result[code] = {"disabled": True}
            continue

        result[code] = {
            "name"    : row.get("name", ""),
            "address1": row.get("address1", ""),
            "address2": row.get("address2", ""),
            "city"    : row.get("city", ""),
            "state"   : row.get("state", ""),
            "zip"     : row.get("zip", ""),
            "phone"   : row.get("phone", ""),
            "disabled": False,
        }

    with OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"buyandship": result}, f, allow_unicode=True, sort_keys=False)
    logging.info("✅ 完了 → %s", OUT)

# ──────────
if __name__ == "__main__":
    asyncio.run(main())
