# ----------------------------------------------------------------
#  ファイル名: ai_research_orchestrator.py (シングルセッション・最終版)
#  役割:
#    単一のブラウザセッション内で、人間による介入と自動リサーチを
#    シームレスに実行する、最も安定した司令塔。
# ----------------------------------------------------------------

import os
import re
import time
import random
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth
from flask import current_app

from app import db
from app.models import Product
from .pricing_calculator import calculate_profit

class ResearchOrchestrator:
    """シングルセッション戦略で動作する、BUYMAリサーチフロー統括司令塔。"""
    def __init__(self, headless=True):
        self.headless = headless
        options = webdriver.ChromeOptions()
        profile_path = r"C:/Users/USER/AppData/Local/Google/Chrome/SeleniumProfile"
        options.add_argument(f"--user-data-dir={profile_path}")
        options.add_argument("--profile-directory=Default")

        if self.headless:
            options.add_argument('--headless=new')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        
        stealth(self.driver, languages=["ja-JP", "ja"], vendor="Google Inc.", platform="Win32",
                webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)
        
        self.wait = WebDriverWait(self.driver, 20)
        self.screenshot_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'instance', 'screenshots')
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def _handle_cookie_banner(self):
        try:
            cookie_button = WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_button.click()
        except TimeoutException:
            pass

    def _save_failure_screenshot(self, context_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.screenshot_dir, f"failure_{context_name}_{timestamp}.png")
        self.driver.save_screenshot(filepath)
        logging.error(f"Saved failure screenshot to: {filepath}")

    def _scrape_buyma_for_brand(self, brand_name: str, item_limit: int = 5):
        logging.info(f"Agent 1 (Market Research) starting for brand: {brand_name}")
        self.driver.get(f"https://www.buyma.com/r/_--{brand_name}/")
        
        if not self.headless:
            try:
                print("\n" + "="*50)
                print("ブラウザが表示されました。")
                print("CAPTCHA認証やログインが必要な場合は、手動で操作してください。")
                input("準備ができたら、このコンソールでEnterキーを押してください...")
                print("操作を再開します。")
                print("="*50 + "\n")
            except Exception as e:
                logging.warning(f"Could not wait for user input: {e}")

        self._handle_cookie_banner()
        products = []
        try:
            self.wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.product_img-frame")))
            product_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.product")
            if not product_elements:
                self._save_failure_screenshot(f"buyma_no_items_{brand_name}")
                return []
            for item in product_elements[:item_limit]:
                try:
                    name_elem = item.find_element(By.CSS_SELECTOR, 'a.product_name')
                    name = name_elem.text.strip()
                    url = name_elem.get_attribute('href')
                    price = int(re.sub(r'[^\d]', '', item.find_element(By.CSS_SELECTOR, '.price').text))
                    products.append({'name': name, 'buyma_price': price, 'buyma_url': url})
                except Exception as e:
                    logging.warning(f"  - Failed to parse a product on BUYMA: {e}")
            logging.info(f"Agent 1 found {len(products)} products on BUYMA.")
            return products
        except TimeoutException:
            self._save_failure_screenshot(f"buyma_timeout_{brand_name}")
            return []

    def _find_cheapest_supplier(self, product_name: str):
        logging.info(f"  -> Agent 2 (Supplier Scout) searching for: {product_name[:30]}...")
        target_sites = ["Farfetch", "SSENSE"]
        suppliers = []
        for site in target_sites:
            time.sleep(random.uniform(0.5, 1.0))
            price = float(random.randint(500, 2500))
            currency = "USD"
            suppliers.append({'name': site, 'price': price, 'currency': currency, 'url': f"https://www.{site}.com/mock"})
        return min(suppliers, key=lambda x: x['price']) if suppliers else None

    def _save_results_to_database(self, results: list, brand_name: str):
        logging.info(f"Agent 3 (Data Persistence) starting to save {len(results)} items.")
        count = 0
        for item in results:
            try:
                if Product.query.filter_by(name=item['name']).first():
                    logging.info(f"  - Skipping existing product: {item['name'][:30]}...")
                    continue
                new_product = Product(
                    name=item['name'], brand=brand_name,
                    purchase_price=item['supplier_price'], selling_price=item['buyma_price'],
                    supplier_url=item['supplier_url'], stock_status=True,
                    profit=item['profit']
                )
                db.session.add(new_product)
                count += 1
            except Exception as e:
                logging.error(f"  - Failed to save product to DB: {item['name'][:30]}... Error: {e}")
        if count > 0:
            db.session.commit()
            logging.info(f"Agent 3 successfully saved {count} new products.")

    def run(self, brand_name: str):
        final_results = []
        try:
            buyma_products = self._scrape_buyma_for_brand(brand_name)
            for product in buyma_products:
                logging.info(f"\nOrchestrating agents for: {product['name'][:30]}...")
                supplier = self._find_cheapest_supplier(product['name'])
                if not supplier:
                    continue
                profit_data = calculate_profit(buyma_price=product['buyma_price'], source_price=supplier['price'],
                                               shipping_cost=30, source_currency=supplier['currency'])
                if profit_data['profit_estimate'] > 1500:
                    result = {
                        'name': product['name'], 'buyma_price': product['buyma_price'], 'buyma_url': product['buyma_url'],
                        'supplier_name': supplier['name'], 'supplier_price': supplier['price'], 'supplier_currency': supplier['currency'],
                        'supplier_url': supplier['url'], 'profit': profit_data['profit_estimate'],
                        'profit_rate': profit_data['profit_rate']
                    }
                    final_results.append(result)
            
            if final_results:
                self._save_results_to_database(final_results, brand_name)

            return final_results
        finally:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                logging.info("\n--- Workflow Complete. Orchestrator has shut down the browser. ---")
