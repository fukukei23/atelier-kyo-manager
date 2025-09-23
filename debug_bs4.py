# debug_bs4.py
"""
完全版：Selenium でページ取得 → BeautifulSoup で価格診断
クラス名がハッシュ化されていても「部分一致」で拾えるように済
"""

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import time
from pathlib import Path

# ---------- ユーザー設定 ----------
URL = "https://www.buyma.com/item/113145560/"  # 調査対象商品ページ
SAVE_FILE = "sample.html"                      # 保存ファイル名
# ----------------------------------

# 価格・参考価格らしきセレクタ（部分一致でハッシュ対応）
PRICE_SELS = [
    "span.price_txt",          # 販売価格
    "strike",                  # 参考価格
    ".percent_box",            # 割引率
]

# 正規表現
YEN_RE = re.compile(r"[¥]\s*([\d,]+)|([\d,]+)\s*円")
PCT_RE = re.compile(
    r"(?:(\d{1,3}))\s*%\s*OFF|(?:OFF\s*(\d{1,3})\s*%)|(?:割引|値引).*?(\d{1,3})\s*%|(?:SALE).*?(\d{1,3})\s*%",
    re.I
)

# ユーティリティ
def _to_int(s: str) -> int | None:
    if not s:
        return None
    try:
        return int(re.sub(r"[^\d]", "", s))
    except ValueError:
        return None

def _first_pct(text: str) -> int | None:
    m = PCT_RE.search(text)
    if not m:
        return None
    for g in m.groups():
        if g and g.isdigit():
            return int(g)
    return None

# ---------- Selenium ページ取得 ----------
def fetch_html() -> str:
    opt = Options()
    opt.add_argument("--headless=new")
    opt.add_argument("--disable-gpu")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-dev-shm-usage")
    opt.add_argument("--disable-blink-features=AutomationControlled")
    opt.add_experimental_option("excludeSwitches", ["enable-automation"])
    opt.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opt)
    try:
        driver.get(URL)
        # 価格ブロックが来るまで待つ（最長10秒）※見つからなくても例外は出さない
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "span[class^='Price_Txt'], [id^='price']"))
            )
        except Exception:
            print("⚠️ 価格ブロックの出現を待機できませんでしたが、強制保存します")
        time.sleep(2)  # JS レンダリング保険
        html = driver.page_source

        # ファイル保存
        Path(SAVE_FILE).write_text(html, encoding="utf-8")
        print(f"✅ HTML を {SAVE_FILE} に保存しました")
        return html
    finally:
        driver.quit()

# ---------- BeautifulSoup 診断 ----------
def diagnose(html: str):
    soup = BeautifulSoup(html, "lxml")
    print("=== BeautifulSoup セレクタ診断（部分一致） ===")
    for sel in PRICE_SELS:
        el = soup.select_one(sel)
        print(f"{sel:35} -> {el}")
        if el:
            print(f"     text='{el.get_text(strip=True)}'")

    print("\n=== 正規表現全文スキャン ===")
    txt = soup.get_text(" ", strip=True)
    m = YEN_RE.search(txt)
    print("YEN_RE ヒット:", m.group(0) if m else "なし")

    # 割引率も拾えるか
    pct = _first_pct(txt)
    print("PCT_RE ヒット:", f"{pct}%") if pct else print("PCT_RE ヒット: なし")

# ---------- メイン ----------
def main():
    html = fetch_html()
    diagnose(html)

if __name__ == "__main__":
    main()
