"""
pytest 設定とフック
- テスト結果を自動的にファイルに保存
- ルートテスト共通フィクスチャ（app/client/auth_client）
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pytest

from app import create_app
from app.extensions import db
from app.models.user import User


# ---------------------------------------------------------------------------
# ルートテスト共通フィクスチャ
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def _env(monkeypatch):
    """create_app() より前に環境変数を設定し、Flask-SQLAlchemy が :memory: DB を使うようにする"""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("AK_STAGE", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")


@pytest.fixture(scope="function")
def routes_app(_env):
    """ルートテスト用 Flask アプリ（in-memory SQLite）"""
    application = create_app()
    application.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
        "CELERY_ALWAYS_EAGER": True,
    })
    with application.app_context():
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def routes_client(routes_app):
    """ルートテスト用 Flask テストクライアント"""
    return routes_app.test_client()


@pytest.fixture()
def routes_test_user(routes_app):
    """ルートテスト用ユーザー（ログイン用）"""
    with routes_app.app_context():
        u = User(username="testuser", display_name="Test", is_active=True)
        u.set_password("password123")
        db.session.add(u)
        db.session.commit()
    return {"username": "testuser", "password": "password123"}


@pytest.fixture()
def routes_auth_client(routes_client, routes_app, routes_test_user):
    """認証済みルートテスト用クライアント"""
    routes_client.post("/auth/login", data=routes_test_user, follow_redirects=False)
    return routes_client


@pytest.fixture(scope="function")
def event_loop():
    """pytest-asyncio互換のevent_loopフィクスチャ"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """pytest設定時に実行されるフック"""
    # プロジェクトルートを取得
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # テスト結果ファイルのパスを設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = reports_dir / f"TEST_RESULTS_{timestamp}.txt"
    config._test_result_file = result_file

    # テスト結果を収集するためのリストを初期化
    config._test_results = []


def pytest_sessionstart(session):
    """テストセッション開始時に実行されるフック"""
    # 開始時刻を記録（pytest_configureより後）
    session.config._test_start_time = datetime.now()


def pytest_runtest_logreport(report):
    """各テストの結果を記録"""
    # テスト結果を収集（callフェーズのみ）
    if report.when == "call":
        # report オブジェクトから config を安全に取得
        # pytest のバージョンによって report.config がない場合があるため、複数の方法を試す
        config = None
        if hasattr(report, "config"):
            config = report.config
        elif hasattr(report, "node") and hasattr(report.node, "config"):
            config = report.node.config

        if config is None or not hasattr(config, "_test_results"):
            # config が取得できない場合はスキップ
            return

        if not hasattr(config, "_test_results"):
            config._test_results = []

        nodeid = getattr(report, "nodeid", "unknown")
        outcome = getattr(report, "outcome", "unknown")

        longrepr_text = None
        if hasattr(report, "longreprtext") and report.longreprtext:
            longrepr_text = report.longreprtext
        elif hasattr(report, "longrepr") and report.longrepr:
            # longreprtextが使えない場合はlongreprを文字列化
            try:
                longrepr_text = str(report.longrepr)
            except Exception:
                longrepr_text = None

        config._test_results.append(
            {
                "nodeid": nodeid,
                "outcome": outcome,  # passed, failed, skipped
                "longrepr": longrepr_text,
            }
        )


def pytest_sessionfinish(session, exitstatus):
    """テストセッション終了時に実行されるフック"""
    if not hasattr(session.config, "_test_result_file"):
        return

    result_file = session.config._test_result_file
    start_time = getattr(session.config, "_test_start_time", datetime.now())
    end_time = datetime.now()
    duration = max(0.0, (end_time - start_time).total_seconds())  # 負の値を防ぐ

    # テスト結果を集計（callフェーズのみ収集済み）
    test_results = getattr(session.config, "_test_results", [])

    total_tests = len(test_results)
    passed = len([r for r in test_results if r.get("outcome") == "passed"])
    failed = len([r for r in test_results if r.get("outcome") == "failed"])
    skipped = len([r for r in test_results if r.get("outcome") == "skipped"])

    # 収集されたテスト数も記録
    collected = getattr(session, "testscollected", total_tests)

    # 結果をファイルに書き込む
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("テスト実行結果\n")
            f.write(f"実行日時: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"終了日時: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"実行時間: {duration:.2f}秒\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"収集されたテスト数: {collected}\n")
            f.write(f"実行されたテスト数: {total_tests}\n")
            f.write(f"✅ 成功: {passed}\n")
            f.write(f"❌ 失敗: {failed}\n")
            f.write(f"⏭️  スキップ: {skipped}\n")
            f.write(f"終了コード: {exitstatus}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            # 失敗したテストの詳細
            if failed > 0:
                f.write("失敗したテスト:\n")
                f.write("-" * 80 + "\n")
                for result in test_results:
                    if result.get("outcome") == "failed":
                        f.write(f"\n{result.get('nodeid', 'unknown')}\n")
                        if result.get("longrepr"):
                            f.write(f"{result.get('longrepr')}\n")
                        f.write("-" * 80 + "\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("詳細なログは pytest の出力を参照してください。\n")
            f.write(f"このファイル: {result_file}\n")
            f.write("=" * 80 + "\n")

        print(f"\n✅ テスト結果を保存しました: {result_file}")

    except Exception as e:
        print(f"\n⚠️ テスト結果ファイルの保存に失敗しました: {e}", file=sys.stderr)
