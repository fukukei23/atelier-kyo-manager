"""
Config/Utils テストカバレッジ
- config/config.py (AppConfig)
- config/constants.py (ビジネス定数)
- config/cost_table.py (送料テーブル)
- config/paths.py (パス定義)
- utils/validation.py (バリデーション)
- utils/errors.py (エラー処理)
- utils/fx_utils.py (為替ユーティリティ 追加テスト)
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app import create_app
from app.extensions import db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def _env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("AK_STAGE", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")


@pytest.fixture(scope="function")
def app(_env):
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
        }
    )
    with app.app_context():
        yield app
        db.session.remove()
        db.drop_all()


# ===========================================================================
# AppConfig
# ===========================================================================
class TestAppConfig:
    def test_config_has_secret_key(self, app):
        assert app.config["SECRET_KEY"] is not None

    def test_config_has_sqlalchemy_uri(self, app):
        assert "SQLALCHEMY_DATABASE_URI" in app.config

    def test_config_track_modifications_false(self, app):
        assert app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] is False

    def test_get_flask_config_returns_dict(self):
        from app.config.config import AppConfig

        config = AppConfig.get_flask_config()
        assert isinstance(config, dict)
        assert "SECRET_KEY" in config
        assert "SQLALCHEMY_DATABASE_URI" in config

    def test_config_has_slack_webhook(self):
        from app.config.config import AppConfig

        config = AppConfig.get_flask_config()
        assert "SLACK_WEBHOOK_URL" in config

    def test_root_dir_is_path(self):
        from app.config.config import AppConfig

        assert isinstance(AppConfig.ROOT_DIR, Path)

    def test_app_dir_is_path(self):
        from app.config.config import AppConfig

        assert isinstance(AppConfig.APP_DIR, Path)

    def test_data_dir_is_path(self):
        from app.config.config import AppConfig

        assert isinstance(AppConfig.DATA_DIR, Path)

    def test_config_dir_is_path(self):
        from app.config.config import AppConfig

        assert isinstance(AppConfig.CONFIG_DIR, Path)

    def test_data_dir_exists(self):
        from app.config.config import AppConfig

        assert AppConfig.DATA_DIR.exists()

    def test_config_dir_exists(self):
        from app.config.config import AppConfig

        assert AppConfig.CONFIG_DIR.exists()

    def test_sites_dir_exists(self):
        from app.config.config import AppConfig

        assert (AppConfig.CONFIG_DIR / "sites").exists()

    def test_get_db_url_default(self):
        from app.config.config import AppConfig

        url = AppConfig.get_db_url()
        assert "sqlite" in url

    def test_get_db_url_from_env(self):
        from app.config.config import AppConfig

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
            assert AppConfig.get_db_url() == "postgresql://test"

    def test_config_alias(self):
        from app.config.config import AppConfig, Config

        assert Config is AppConfig

    def test_browser_enum(self):
        from app.config.config import Browser

        assert Browser.CHROME == "chrome"
        assert Browser.EDGE == "edge"

    def test_log_level_enum(self):
        from app.config.config import LogLevel

        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"

    def test_sqlite_uri(self):
        from app.config.config import _sqlite_uri

        result = _sqlite_uri(Path("/tmp/test.db"))
        assert result.startswith("sqlite:///")
        assert "test.db" in result

    def test_sites_config_is_dict(self):
        from app.config.config import AppConfig

        assert isinstance(AppConfig.SITES, dict)

    def test_load_layered_site_config_no_file(self, tmp_path):
        from app.config.config import _load_layered_site_config

        result = _load_layered_site_config(tmp_path)
        assert result == {}

    def test_load_layered_site_config_with_file(self, tmp_path):
        import json

        from app.config.config import _load_layered_site_config

        sites_dir = tmp_path / "sites"
        sites_dir.mkdir()
        base_file = sites_dir / "base.json"
        base_file.write_text(json.dumps({"sites": [{"name": "test-site", "url": "https://example.com"}]}))
        result = _load_layered_site_config(tmp_path)
        assert "test-site" in result
        assert result["test-site"]["url"] == "https://example.com"


# ===========================================================================
# Constants
# ===========================================================================
class TestConstants:
    def test_commission_rates(self):
        from app.config.constants import DOMESTIC_COMMISSION_RATE, OVERSEAS_COMMISSION_RATE

        assert DOMESTIC_COMMISSION_RATE == 0.077
        assert OVERSEAS_COMMISSION_RATE == 0.055

    def test_platform_fee_rate(self):
        from app.config.constants import PLATFORM_FEE_RATE

        assert PLATFORM_FEE_RATE == 0.077

    def test_effective_fee_rate(self):
        from app.config.constants import EFFECTIVE_FEE_RATE

        assert EFFECTIVE_FEE_RATE == 0.142

    def test_transfer_fee(self):
        from app.config.constants import TRANSFER_FEE

        assert TRANSFER_FEE == 220.0

    def test_default_exchange_rate(self):
        from app.config.constants import DEFAULT_EXCHANGE_RATE_USDJPY

        assert DEFAULT_EXCHANGE_RATE_USDJPY == 150.0

    def test_payment_method_extension_days(self):
        from app.config.constants import PAYMENT_METHOD_EXTENSION_DAYS

        assert isinstance(PAYMENT_METHOD_EXTENSION_DAYS, dict)
        assert PAYMENT_METHOD_EXTENSION_DAYS["credit_card"] == 45
        assert PAYMENT_METHOD_EXTENSION_DAYS["bank_transfer"] == 90

    def test_expected_payment_days(self):
        from app.config.constants import EXPECTED_PAYMENT_DAYS

        assert EXPECTED_PAYMENT_DAYS["domestic"] == 15
        assert EXPECTED_PAYMENT_DAYS["overseas"] == 25

    def test_max_lengths(self):
        from app.config.constants import MAX_MESSAGE_LENGTH, MAX_SUBJECT_LENGTH

        assert MAX_SUBJECT_LENGTH == 200
        assert MAX_MESSAGE_LENGTH == 1000

    def test_scraper_constants(self):
        from app.config.constants import SCRAPER_CACHE_SIZE, SCRAPER_MAX_RETRIES, SCRAPER_REQUEST_TIMEOUT

        assert SCRAPER_CACHE_SIZE == 100
        assert SCRAPER_REQUEST_TIMEOUT == 10
        assert SCRAPER_MAX_RETRIES == 3

    def test_fx_constants(self):
        from app.config.constants import FX_CACHE_TTL_HOURS, FX_REQUEST_TIMEOUT

        assert FX_REQUEST_TIMEOUT == 10
        assert FX_CACHE_TTL_HOURS == 12

    def test_llm_constants(self):
        from app.config.constants import LLM_CACHE_TTL_SECONDS, LLM_RETRY_BACKOFF_BASE

        assert LLM_RETRY_BACKOFF_BASE == 1.5
        assert LLM_CACHE_TTL_SECONDS > 0

    def test_payment_methods_all_have_positive_days(self):
        from app.config.constants import PAYMENT_METHOD_EXTENSION_DAYS

        for method, days in PAYMENT_METHOD_EXTENSION_DAYS.items():
            assert days > 0, f"{method} has non-positive extension days: {days}"


# ===========================================================================
# Cost Table
# ===========================================================================
class TestCostTable:
    def test_buyandship_shipping_rates(self):
        from app.config.cost_table import BUYANDSHIP_SHIPPING_RATES

        assert "light" in BUYANDSHIP_SHIPPING_RATES
        assert "medium" in BUYANDSHIP_SHIPPING_RATES
        assert "heavy" in BUYANDSHIP_SHIPPING_RATES
        assert "default" in BUYANDSHIP_SHIPPING_RATES

    def test_light_is_cheapest(self):
        from app.config.cost_table import BUYANDSHIP_SHIPPING_RATES

        assert BUYANDSHIP_SHIPPING_RATES["light"] <= BUYANDSHIP_SHIPPING_RATES["medium"]

    def test_heavy_is_most_expensive(self):
        from app.config.cost_table import BUYANDSHIP_SHIPPING_RATES

        assert BUYANDSHIP_SHIPPING_RATES["heavy"] >= BUYANDSHIP_SHIPPING_RATES["medium"]

    def test_category_weight_class(self):
        from app.config.cost_table import CATEGORY_WEIGHT_CLASS

        assert CATEGORY_WEIGHT_CLASS["wallet"] == "light"
        assert CATEGORY_WEIGHT_CLASS["bag"] == "medium"
        assert CATEGORY_WEIGHT_CLASS["shoes"] == "medium"

    def test_default_markup_rate(self):
        from app.config.cost_table import DEFAULT_MARKUP_RATE

        assert DEFAULT_MARKUP_RATE == 1.3

    def test_get_buyandship_shipping_with_category(self):
        from app.config.cost_table import get_buyandship_shipping

        assert get_buyandship_shipping("wallet") == 3300.0
        assert get_buyandship_shipping("bag") == 3300.0
        assert get_buyandship_shipping("clothing") == 3300.0

    def test_get_buyandship_shipping_unknown_category(self):
        from app.config.cost_table import get_buyandship_shipping

        assert get_buyandship_shipping("unknown") == 3300.0

    def test_get_buyandship_shipping_none_category(self):
        from app.config.cost_table import get_buyandship_shipping

        assert get_buyandship_shipping(None) == 3300.0

    def test_get_buyandship_shipping_heavy(self):
        from app.config.cost_table import get_buyandship_shipping

        # heavy is not directly mapped in CATEGORY_WEIGHT_CLASS but exists in rates
        assert get_buyandship_shipping() == 3300.0

    def test_all_shipping_rates_positive(self):
        from app.config.cost_table import BUYANDSHIP_SHIPPING_RATES

        for key, rate in BUYANDSHIP_SHIPPING_RATES.items():
            assert rate > 0, f"{key} has non-positive rate: {rate}"

    def test_all_category_weight_classes_valid(self):
        from app.config.cost_table import BUYANDSHIP_SHIPPING_RATES, CATEGORY_WEIGHT_CLASS

        for cat, weight_class in CATEGORY_WEIGHT_CLASS.items():
            assert weight_class in BUYANDSHIP_SHIPPING_RATES, f"{cat} maps to invalid weight class: {weight_class}"


# ===========================================================================
# Paths
# ===========================================================================
class TestPaths:
    def test_project_root_is_path(self):
        from app.config.paths import PROJECT_ROOT

        assert isinstance(PROJECT_ROOT, Path)

    def test_project_root_exists(self):
        from app.config.paths import PROJECT_ROOT

        assert PROJECT_ROOT.exists()

    def test_scripts_dev_dir(self):
        from app.config.paths import SCRIPTS_DEV_DIR

        assert "scripts" in str(SCRIPTS_DEV_DIR)
        assert "dev" in str(SCRIPTS_DEV_DIR)

    def test_docs_reports_dir(self):
        from app.config.paths import DOCS_REPORTS_DIR

        assert "docs" in str(DOCS_REPORTS_DIR)
        assert "reports" in str(DOCS_REPORTS_DIR)

    def test_logs_dir(self):
        from app.config.paths import LOGS_GENERATED_DIR

        assert "logs" in str(LOGS_GENERATED_DIR)

    def test_tmp_dir(self):
        from app.config.paths import TMP_GENERATED_DIR

        assert "tmp" in str(TMP_GENERATED_DIR)

    def test_data_generated_dir(self):
        from app.config.paths import DATA_GENERATED_DIR

        assert "data" in str(DATA_GENERATED_DIR)

    def test_instance_dir(self):
        from app.config.paths import INSTANCE_DIR

        assert "instance" in str(INSTANCE_DIR)

    def test_output_dir(self):
        from app.config.paths import OUTPUT_DIR

        assert "output" in str(OUTPUT_DIR)

    def test_exports_dir(self):
        from app.config.paths import EXPORTS_DIR

        assert "exports" in str(EXPORTS_DIR)

    def test_ensure_dirs(self):
        from app.config.paths import ensure_dirs

        # Should not raise
        ensure_dirs()

    def test_all_dirs_are_under_project_root(self):
        from app.config.paths import (
            DATA_GENERATED_DIR,
            DOCS_REPORTS_DIR,
            LOGS_GENERATED_DIR,
            PROJECT_ROOT,
            SCRIPTS_DEV_DIR,
            TMP_GENERATED_DIR,
        )

        for d in [SCRIPTS_DEV_DIR, DOCS_REPORTS_DIR, LOGS_GENERATED_DIR, TMP_GENERATED_DIR, DATA_GENERATED_DIR]:
            # Check the directory starts with PROJECT_ROOT path
            assert str(d).startswith(str(PROJECT_ROOT)), f"{d} is not under PROJECT_ROOT"


# ===========================================================================
# Validation
# ===========================================================================
class TestValidation:
    def test_validate_url_https(self):
        from app.utils.validation import validate_url

        assert validate_url("https://example.com") == "https://example.com"

    def test_validate_url_http_rejected(self):
        from app.utils.validation import ValidationError, validate_url

        with pytest.raises(ValidationError, match="許可されていないスキーム"):
            validate_url("http://example.com")

    def test_validate_url_no_host_rejected(self):
        from app.utils.validation import ValidationError, validate_url

        with pytest.raises(ValidationError, match="ホスト"):
            validate_url("https://")

    def test_validate_url_custom_scheme(self):
        from app.utils.validation import validate_url

        assert validate_url("ftp://files.com", allowed_schemes=("ftp",)) == "ftp://files.com"

    def test_validate_int_valid(self):
        from app.utils.validation import validate_int

        assert validate_int("42") == 42

    def test_validate_int_with_string(self):
        from app.utils.validation import validate_int

        assert validate_int(42) == 42

    def test_validate_int_invalid(self):
        from app.utils.validation import ValidationError, validate_int

        with pytest.raises(ValidationError, match="整数"):
            validate_int("abc")

    def test_validate_int_none(self):
        from app.utils.validation import ValidationError, validate_int

        with pytest.raises(ValidationError):
            validate_int(None)

    def test_validate_int_min_val(self):
        from app.utils.validation import validate_int

        assert validate_int("5", min_val=1) == 5

    def test_validate_int_below_min(self):
        from app.utils.validation import ValidationError, validate_int

        with pytest.raises(ValidationError, match="以上"):
            validate_int("0", min_val=1)

    def test_validate_int_max_val(self):
        from app.utils.validation import validate_int

        assert validate_int("5", max_val=10) == 5

    def test_validate_int_above_max(self):
        from app.utils.validation import ValidationError, validate_int

        with pytest.raises(ValidationError, match="以下"):
            validate_int("15", max_val=10)

    def test_validate_int_with_name(self):
        from app.utils.validation import ValidationError, validate_int

        with pytest.raises(ValidationError, match="page"):
            validate_int("abc", name="page")

    def test_validate_float_valid(self):
        from app.utils.validation import validate_float

        assert validate_float("3.14") == pytest.approx(3.14)

    def test_validate_float_with_number(self):
        from app.utils.validation import validate_float

        assert validate_float(3.14) == pytest.approx(3.14)

    def test_validate_float_invalid(self):
        from app.utils.validation import ValidationError, validate_float

        with pytest.raises(ValidationError, match="数値"):
            validate_float("abc")

    def test_validate_float_min_val(self):
        from app.utils.validation import validate_float

        assert validate_float("1.5", min_val=0.0) == pytest.approx(1.5)

    def test_validate_float_below_min(self):
        from app.utils.validation import ValidationError, validate_float

        with pytest.raises(ValidationError, match="以上"):
            validate_float("-1.0", min_val=0.0)

    def test_validate_float_max_val(self):
        from app.utils.validation import validate_float

        assert validate_float("5.0", max_val=10.0) == pytest.approx(5.0)

    def test_validate_float_above_max(self):
        from app.utils.validation import ValidationError, validate_float

        with pytest.raises(ValidationError, match="以下"):
            validate_float("15.0", max_val=10.0)

    def test_sanitize_filename_normal(self):
        from app.utils.validation import sanitize_filename

        assert sanitize_filename("test.txt") == "test.txt"

    def test_sanitize_filename_path_traversal(self):
        from app.utils.validation import sanitize_filename

        assert sanitize_filename("../../etc/passwd") == "passwd"

    def test_sanitize_filename_absolute_path(self):
        from app.utils.validation import sanitize_filename

        assert sanitize_filename("/tmp/secret.txt") == "secret.txt"

    def test_sanitize_filename_current_dir(self):
        from app.utils.validation import sanitize_filename

        assert sanitize_filename("./config.ini") == "config.ini"

    def test_validation_error_is_value_error(self):
        from app.utils.validation import ValidationError

        assert issubclass(ValidationError, ValueError)


# ===========================================================================
# Errors (safe_error_msg)
# ===========================================================================
class TestErrors:
    def test_safe_error_msg_test_stage(self, app):
        from app.utils.errors import safe_error_msg

        msg = safe_error_msg(ValueError("test error"), context="test_ctx")
        assert "test error" in msg
        assert "test_ctx" in msg

    def test_safe_error_msg_no_context(self, app):
        from app.utils.errors import safe_error_msg

        msg = safe_error_msg(ValueError("some error"))
        assert "some error" in msg

    def test_safe_error_msg_prod_stage(self):
        from app import create_app

        test_app = create_app()
        test_app.config["STAGE"] = "prod"
        with test_app.app_context():
            from app.utils.errors import safe_error_msg

            msg = safe_error_msg(ValueError("sensitive info"), context="api")
            assert "sensitive info" not in msg
            assert "エラーが発生" in msg

    def test_safe_error_msg_staging_stage(self):
        from app import create_app

        test_app = create_app()
        test_app.config["STAGE"] = "staging"
        with test_app.app_context():
            from app.utils.errors import safe_error_msg

            msg = safe_error_msg(ValueError("debug info"), context="staging_ctx")
            assert "debug info" in msg


# ===========================================================================
# FX Utils (additional tests beyond existing test_fx_utils.py)
# ===========================================================================
class TestFxUtilsExtras:
    def test_parse_fx_rates_str_single(self):
        from app.utils.fx_utils import parse_fx_rates_str

        result = parse_fx_rates_str("USD=150.0")
        assert result == {"USD": 150.0}

    def test_parse_fx_rates_str_empty_value(self):
        from app.utils.fx_utils import parse_fx_rates_str

        result = parse_fx_rates_str("USD=")
        # Empty string after =, float("") raises ValueError, so it's skipped
        assert result == {}

    def test_build_jpy_table_from_eur_multiple_currencies(self):
        from app.utils.fx_utils import build_jpy_table_from_eur

        eur_table = {"EUR": 1.0, "JPY": 160.0, "USD": 1.1, "GBP": 0.85}
        result = build_jpy_table_from_eur(eur_table)
        assert result["JPY"] == 1.0
        assert result["EUR"] == pytest.approx(160.0)
        assert result["USD"] == pytest.approx(160.0 / 1.1)
        assert result["GBP"] == pytest.approx(160.0 / 0.85)

    def test_ecb_xml_to_eur_table_multiple_currencies(self):
        from app.utils.fx_utils import _ecb_xml_to_eur_table

        xml = """<root>
            <Cube time="2024-01-01">
                <Cube currency="USD" rate="1.1"/>
                <Cube currency="JPY" rate="160.0"/>
                <Cube currency="GBP" rate="0.85"/>
            </Cube>
        </root>"""
        result = _ecb_xml_to_eur_table(xml)
        assert result["EUR"] == 1.0
        assert result["USD"] == 1.1
        assert result["JPY"] == 160.0
        assert result["GBP"] == 0.85

    def test_get_fx_table_jpy_manual(self):
        from app.utils.fx_utils import get_fx_table_jpy

        manual = {"USD": 150.0, "EUR": 165.0}
        table, meta = get_fx_table_jpy(manual_table=manual, auto=False)
        assert table["USD"] == 150.0
        assert meta["source"] == "manual"
        assert meta["cache"] is False

    def test_get_fx_table_jpy_no_source(self):
        from app.utils.fx_utils import get_fx_table_jpy

        table, meta = get_fx_table_jpy(manual_table=None, auto=False)
        assert table == {}
        assert meta["source"] is None

    def test_get_fx_table_jpy_manual_case_insensitive(self):
        from app.utils.fx_utils import get_fx_table_jpy

        manual = {"usd": 150.0, "eur": 165.0}
        table, meta = get_fx_table_jpy(manual_table=manual, auto=False)
        assert "USD" in table
        assert "EUR" in table
