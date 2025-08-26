# ----------------------------------------------------------------
# ファイル名: app/utils/ai_research_orchestrator.py
# バージョン: 3.7 (2025-08-26 JST)
# 目的:
#   BUYMA一覧→(ブランド優先抽出)→仕入先探索(SupplierScout)→利益計算→保存/出力
# 変更点:
#   - 一覧カードの親要素取得を強化し価格文字列を高精度に抽出
#   - 商品詳細ページの価格抽出を JS + JSON-LD で堅牢化（:contains 廃止）
#   - 抽出失敗時も安全にフォールバック、証跡を保存
# 使い方(例):
#   python -c "from app.utils.ai_research_orchestrator import ResearchOrchestrator as R; o=R(headless=False); print(o.run('GUCCI'))"
# 出力:
#   output/research_<BRAND>_<TS>.json / .csv
#   instance/screenshots/failure_*.png
# ----------------------------------------------------------------
from __future__ import annotations

import os, re, time, logging, json, csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth

from app import db
from app.models import Product
from .pricing_calculator import calculate_profit
from .ai_supplier_scout import SupplierScout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ResearchOrchestrator:
    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        options = webdriver.ChromeOptions()
        profile_path = r"C:/Users/USER/AppData/Local/Google/Chrome/SeleniumProfile"
        options.add_argument(f"--user-data-dir={profile_path}")
        options.add_argument("--profile-directory=Default")
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        stealth(self.driver, languages=["ja-JP","ja"], vendor="Google Inc.", platform="Win32",
                webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)
        self.wait = WebDriverWait(self.driver, 20)
        self.screenshot_dir = Path(__file__).resolve().parents[2] / "instance" / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.supplier_scout = SupplierScout(headless=self.headless)

    # ------------ Utils ------------
    def _handle_cookie_banner(self) -> None:
        try:
            btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            btn.click()
        except TimeoutException:
            pass

    def _save_failure_screenshot(self, context_name: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = self.screenshot_dir / f"failure_{context_name}_{ts}.png"
        try:
            self.driver.save_screenshot(str(fp))
            logging.error(f"Saved failure screenshot to: {fp}")
        except Exception as e:
            logging.error(f"Failed to save screenshot: {e}")

    def _normalize_supplier_query(self, raw_name: str) -> str:
        m = re.search(r"\b([A-Za-z]{2,}\d{2,}[A-Za-z]*)\b", raw_name)
        if m:
            return f"Gucci {m.group(1)}"   # 例: GG0637SK
        tokens = re.findall(r"[A-Za-z0-9\-]{3,}", raw_name)
        return "Gucci " + " ".join(tokens[:3]) if tokens else "Gucci"

    # 一覧カードから価格を拾うためのJS（カードの祖先を広めに取り、テキストから¥/円を抽出）
    def _js_collect_item_anchors(self):
        script = """
        const out = [];
        const anchors = Array.from(document.querySelectorAll("a[href*='/item/']"));
        const yenRe = /[¥\\u00A5]\\s*([\\d,]+)|([\\d,]+)\\s*円/;
        function text(e){ return (e && (e.innerText || e.textContent) || "").replace(/\\s+/g," ").trim(); }
        function findPriceText(root){
          if(!root) return "";
          // カード内の候補ノードを走査（最大150ノード）
          const nodes = [root, ...Array.from(root.querySelectorAll("*")).slice(0,150)];
          for(const n of nodes){
            const t = text(n);
            if(yenRe.test(t)) return t;
          }
          return "";
        }
        for (const a of anchors){
          try{
            let root = a.closest("li, article, div[class*='product'], div[class*='card']") || a.parentElement;
            const cardText = text(root);
            const priceText = findPriceText(root) || cardText;
            const href = a.getAttribute('href');
            const url = href && href.startsWith('http') ? href : (location.origin + (href || ""));
            const name = text(a) || text(root).slice(0,180);
            out.push({url, name, priceText});
          }catch(e){}
        }
        return out;
        """
        return self.driver.execute_script(script)

    # 商品詳細ページで価格を抽出（CSS候補 → 全ノード → JSON-LD）
    def _extract_price_from_product_page(self) -> Optional[int]:
        js = """
        const yenRe = /[¥\\u00A5]\\s*([\\d,]+)|([\\d,]+)\\s*円/;
        const sel = [
          "[itemprop='price']",
          "meta[itemprop='price']",
          "[data-test*='price' i]",
          "[class*='price' i]",
          "[id*='price' i]"
        ];
        function pickInt(s){
          if(!s) return "";
          const m = s.match(/[\\d,]+/);
          return m ? m[0] : "";
        }
        function txt(e){
          if(!e) return "";
          if(e.tagName === "META" && e.content) return e.content;
          return (e.innerText || e.textContent || "").trim();
        }
        try{
          // 1) CSS候補
          for(const s of sel){
            const els = Array.from(document.querySelectorAll(s)).slice(0,50);
            for(const el of els){
              const t = txt(el);
              if(yenRe.test(t)) return pickInt(t);
              if(/^[\\d,.]+$/.test(t)) return pickInt(t);
            }
          }
          // 2) 全ノード走査（上限1500）
          const all = Array.from(document.querySelectorAll("body *")).slice(0,1500);
          for(const el of all){
            const t = txt(el);
            if(yenRe.test(t)) return pickInt(t);
          }
          // 3) JSON-LD offers.price
          const scripts = Array.from(document.querySelectorAll("script[type='application/ld+json']"));
          const scan = (node) => {
            if(!node) return null;
            if(Array.isArray(node)){
              for(const n of node){ const r = scan(n); if(r) return r; }
              return null;
            }
            if(typeof node === "object"){
              if(node["@type"] === "Product"){
                const offers = node["offers"];
                const pick = (o) => (o && (o.price || (o.priceSpecification && o.priceSpecification.price)));
                if(Array.isArray(offers)){
                  for(const o of offers){ const p = pick(o); if(p) return String(p); }
                }else{
                  const p = pick(offers); if(p) return String(p);
                }
              }
              for(const k in node){ const r = scan(node[k]); if(r) return r; }
            }
            return null;
          };
          for(const sc of scripts){
            try{
              const data = JSON.parse(sc.textContent);
              const p = scan(data);
              if(p) return pickInt(String(p));
            }catch(e){}
          }
          return "";
        }catch(e){ return ""; }
        """
        try:
            s = self.driver.execute_script(js)
            if s:
                return int(str(s).replace(",", ""))
        except Exception:
            return None
        return None

    # ------------ Agent1: BUYMA抽出 ------------
    def _scrape_buyma_for_brand(self, brand_name: str, item_limit: int = 5) -> List[Dict]:
        logging.info(f"Agent 1 (Market Research) starting for brand: {brand_name}")
        search_url = f"https://www.buyma.com/r/?kw={re.sub(r'\\s+', '+', brand_name)}"
        self.driver.get(search_url)

        if not self.headless:
            try:
                print("\n" + "="*50)
                print("ブラウザが表示されました。必要があればログイン/認証を実施してください。")
                input("準備ができたら、このコンソールでEnterキーを押してください...")
                print("="*50 + "\n")
            except Exception as e:
                logging.warning(f"Could not wait for user input: {e}")

        self._handle_cookie_banner()
        try:
            self.driver.set_window_size(1400, 1200)
        except Exception:
            pass

        anchors = []
        seen = set()
        for _ in range(6):
            try:
                self.driver.execute_script("window.scrollBy(0, Math.max(900, window.innerHeight*0.9));")
            except Exception:
                pass
            time.sleep(0.8)
            try:
                batch = self._js_collect_item_anchors()
            except Exception:
                batch = []
            for it in batch:
                url = it.get("url") or ""
                if "/item/" not in url or url in seen:
                    continue
                seen.add(url)
                anchors.append(it)
            if len(anchors) >= max(item_limit*3, 15):
                break

        if not anchors:
            self._save_failure_screenshot(f"buyma_timeout_{brand_name}")
            logging.error("No product anchors discovered on BUYMA listing.")
            return []

        # ブランド優先フィルタ（0件なら無視して続行）
        filtered = []
        for it in anchors:
            block_text = (it.get("name","") + " " + it.get("priceText","")).strip()
            u = block_text.upper()
            if ("GUCCI" in u) or ("グッチ" in block_text):
                filtered.append(it)
        candidates = filtered or anchors

        # 一覧テキストから価格を抽出
        results = []
        for it in candidates:
            name = re.sub(r"\s+", " ", (it.get("name") or "")).strip()
            url = it.get("url") or ""
            price_text = it.get("priceText") or ""
            m = re.search(r"[¥\u00A5]\s*([\d,]+)|([\d,]+)\s*円", price_text)
            if m and url:
                try:
                    val = m.group(1) or m.group(2)
                    price = int(val.replace(",", ""))
                    if name and price:
                        results.append({"name": name, "buyma_price": price, "buyma_url": url})
                except Exception:
                    pass

        # 一覧から取れなければ、詳細ページで抽出（堅牢化版）
        if not results:
            logging.info("Cards found but no price parsed; fallback: open top cards to parse price.")
            for it in candidates[:10]:
                url = it["url"]
                try:
                    self.driver.execute_script("window.open(arguments[0], '_blank');", url)
                    self.driver.switch_to.window(self.driver.window_handles[-1])
                    time.sleep(1.2)
                    price = self._extract_price_from_product_page()
                    if price:
                        name = re.sub(r"\s+", " ", (it.get("name") or self.driver.title)).strip()
                        results.append({"name": name, "buyma_price": int(price), "buyma_url": url})
                except Exception:
                    pass
                finally:
                    try:
                        self.driver.close()
                        self.driver.switch_to.window(self.driver.window_handles[0])
                    except Exception:
                        pass

        if not results:
            self._save_failure_screenshot(f"buyma_parse_{brand_name}")
            logging.error("Cards found but failed to parse name/price.")
            return []

        # 既視感排除
        out, dedup = [], set()
        for r in results:
            if r["buyma_url"] in dedup:
                continue
            dedup.add(r["buyma_url"])
            out.append(r)

        logging.info(f"Agent 1 found {len(out)} products on BUYMA (top {item_limit} will be used).")
        return out[:item_limit]

    # ------------ Agent2 ------------
    def _find_cheapest_supplier(self, query: str) -> Optional[Dict]:
        logging.info(f"  -> Agent 2 (Supplier Scout) searching for: {query[:60]}...")
        try:
            best = self.supplier_scout.find_cheapest_supplier(query)
        except KeyboardInterrupt:
            logging.warning("SupplierScout interrupted by user; skip this item.")
            return None
        if best:
            logging.info(f"     Best supplier: {best['site']} {best['price']} {best['currency']} {best['url']}")
        else:
            logging.info("     No supplier found.")
        return best

    # ------------ DB保存/ダンプ ------------
    def _save_results_to_database(self, results: List[Dict], brand_name: str) -> None:
        logging.info(f"Agent 3 (Data Persistence) starting to save {len(results)} items.")
        count = 0
        for item in results:
            try:
                if Product.query.filter_by(name=item["name"]).first():
                    logging.info(f"  - Skipping existing product: {item['name'][:30]}...")
                    continue
                db.session.add(Product(
                    name=item["name"],
                    brand=brand_name,
                    purchase_price=item["supplier_price"],
                    selling_price=item["buyma_price"],
                    supplier_url=item["supplier_url"],
                    stock_status=True,
                    profit=item["profit"],
                ))
                count += 1
            except Exception as e:
                logging.error(f"  - Failed to save product to DB: {item['name'][:30]}... Error: {e}")
        if count > 0:
            db.session.commit()
            logging.info(f"Agent 3 successfully saved {count} new products.")

    def _dump_results(self, brand_name: str, data: List[Dict]) -> None:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (out_dir / f"research_{brand_name}_{ts}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if data:
            keys = ["name","buyma_price","buyma_url","supplier_name","supplier_price",
                    "supplier_currency","supplier_url","profit","profit_rate"]
            with open(out_dir / f"research_{brand_name}_{ts}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in data:
                    w.writerow({k: r.get(k, "") for k in keys})

    # ------------ 実行 ------------
    def run(self, brand_name: str, item_limit: int = 5) -> List[Dict]:
        final_results: List[Dict] = []
        try:
            buyma_products = self._scrape_buyma_for_brand(brand_name, item_limit=item_limit)
            for product in buyma_products:
                logging.info(f"\nOrchestrating agents for: {product['name'][:60]}...")
                query = self._normalize_supplier_query(product["name"])
                logging.info(f"  -> normalized query: {query}")
                supplier = self._find_cheapest_supplier(query)
                if not supplier:
                    continue

                profit_data = calculate_profit(
                    buyma_price=product["buyma_price"],
                    source_price=supplier["price"],
                    shipping_cost=30,
                    source_currency=supplier["currency"],
                )

                if profit_data.get("profit_estimate", 0) > 1500:
                    final_results.append({
                        "name": product["name"],
                        "buyma_price": product["buyma_price"],
                        "buyma_url": product["buyma_url"],
                        "supplier_name": supplier["name"],
                        "supplier_price": supplier["price"],
                        "supplier_currency": supplier["currency"],
                        "supplier_url": supplier["url"],
                        "profit": profit_data["profit_estimate"],
                        "profit_rate": profit_data["profit_rate"],
                    })

            if final_results:
                self._save_results_to_database(final_results, brand_name)
            return final_results
        finally:
            self._dump_results(brand_name, final_results)
            if getattr(self, "driver", None):
                try:
                    self.driver.quit()
                finally:
                    logging.info("\n--- Workflow Complete. Orchestrator has shut down the browser. ---")
