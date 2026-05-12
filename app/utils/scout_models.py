"""サイト設定モデル・デフォルトサイト定義"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SiteSelectors:
    search_open: list[str] = field(default_factory=list)
    search_input: list[str] = field(default_factory=list)
    search_submit: list[str] = field(default_factory=list)
    results_item: str | None = None
    first_product_link: str | None = None
    pdp_title: list[str] = field(default_factory=list)
    pdp_price: list[str] = field(default_factory=list)


@dataclass
class SiteConfig:
    name: str
    home_url: str
    domains: list[str] = field(default_factory=list)
    search_mode: str = "human"
    search_template: str | None = None
    wait_until: str = "domcontentloaded"
    timeout_sec: int = 25
    currency_hint: str | None = None
    selectors: SiteSelectors = field(default_factory=SiteSelectors)
    force_ui_search: bool = False
    notes: str | None = None


def default_sites() -> list[SiteConfig]:
    return [
        SiteConfig(
            name="SSENSE",
            home_url="https://www.ssense.com/ja-jp",
            domains=["ssense.com"],
            search_template="https://www.ssense.com/ja-jp/search?q={q}",
            selectors=SiteSelectors(
                search_open=[
                    "a.mobile-header-search",
                    "i.fa-ssense-magnifier",
                    "a[data-test='mobileNavigationSearchLink']",
                    "button[aria-label='Open Search']",
                ],
                search_input=[
                    "#search-form-input",
                    "input[data-testid='search-input']",
                    "input[type='search']",
                    "input[name='q']",
                ],
                search_submit=["#searchSubmitIcon", "button[type='submit']"],
                results_item="a[href*='/product/']",
                first_product_link="a[href*='/product/']",
                pdp_title=["h1", "h2#pdpProductNameText", ".pdp-product-title__name"],
                pdp_price=[
                    "[data-test='pdpRegularPriceText']",
                    ".product-price__sale",
                    "span:has-text('¥')",
                    "span:has-text('￥')",
                ],
            ),
        ),
        SiteConfig(
            name="BUYMA",
            home_url="https://www.buyma.com/",
            domains=["buyma.com"],
            search_template="https://www.buyma.com/r/-/search/?q={q}",
            selectors=SiteSelectors(
                search_input=["#search_txt", "input.fab-search-txtarea", "input#srchTxt"],
                search_submit=["form#search_form", "button#srchBtn"],
                results_item="a[href*='/item/']",
                first_product_link="a[href*='/item/']",
                pdp_title=["h1[itemprop='name']", "h1.product_title", "h1"],
                pdp_price=["span.Price_Txt", "#price", ".product_price .Price_Txt", "span[itemprop='price']"],
            ),
        ),
        SiteConfig(
            name="FARFETCH",
            home_url="https://www.farfetch.com/",
            domains=["farfetch.com"],
            search_template="https://www.farfetch.com/shopping/men/items.aspx?q={q}",
            selectors=SiteSelectors(
                results_item="a[data-testid='productCard-link'], a[href*='/shopping/']",
                first_product_link="a[data-testid='productCard-link'], a[href*='/shopping/']",
                pdp_title=["h1[data-tstid='product-name']", "h1", "span[itemprop='name']"],
                pdp_price=[
                    "[data-tstid='priceInfo-original']",
                    "[data-tstid='priceInfo-onsale']",
                    "p[data-tstid='priceInfo']",
                    "span[data-tstid='current-price']",
                ],
            ),
        ),
        SiteConfig(
            name="MATCHES",
            home_url="https://www.matchesfashion.com/",
            domains=["matchesfashion.com"],
            search_template="https://www.matchesfashion.com/intl/search?text={q}",
            selectors=SiteSelectors(
                results_item="a[href*='/products/']",
                first_product_link="a[href*='/products/']",
                pdp_title=["h1", "h1[data-test='pdp-title']", "div[data-test='pdp-title']"],
                pdp_price=["span[data-test='pdp-price']", "div.prices span.price", "span.now", "span.was"],
            ),
        ),
        SiteConfig(
            name="GENERIC_PDP",
            home_url="",
            domains=[],
            selectors=SiteSelectors(
                pdp_title=["h1", "title"],
                pdp_price=["meta[itemprop='price']", "span[itemprop='price']", "span:has-text('¥')"],
            ),
            notes="--pdp-url",
        ),
    ]
