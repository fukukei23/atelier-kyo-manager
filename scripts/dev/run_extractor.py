from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from app.extractors.product_info_extractor import extract_product_info

d = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
d.get("https://www.buyma.com/item/113145560/")
WebDriverWait(d, 10).until(lambda x: x.execute_script("return document.readyState") == "complete")

html = d.page_source
info = extract_product_info(html, site="BUYMA")
print(info["selectors_used"])
print(info["raw_texts"])
print(info["source_flags"])

d.quit()
