"""Celery アプリケーション定義.

Flask アプリケーションと連携し、Redis をブローカー/バックエンドとして
非同期タスクを実行する Celery インスタンスを提供する。
"""

from __future__ import annotations

import os


def make_celery(app=None):
    """Flask アプリ連携の Celery インスタンスを生成する."""
    from celery import Celery
    from celery.schedules import crontab

    broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    celery = Celery(
        app.import_name if app else "atelier_kyo",
        broker=broker_url,
        backend=result_backend,
        # app/tasks/*.py を worker 起動時に import してタスクレジストリへ登録。
        # autodiscover_tasks は app/tasks.py を想定する規約で本構造(<name>_tasks.py 複数)では動かないため include 明示。
        include=["app.tasks.scrape_tasks", "app.tasks.monitor_tasks"],
    )
    celery.conf.update(
        result_expires=3600,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        beat_schedule={
            "monitor-prices-every-4-hours": {
                "task": "monitor_prices_periodic",
                "schedule": crontab(minute=0, hour="*/4"),
            },
        },
    )

    return celery


_flask_app = None


def _get_flask_app():
    """タスク実行時まで create_app() の生成を遅延する（singleton）.

    celery_app は app/__init__.py → extensions.py:34 から import されるため、
    import 時に create_app() を呼ぶと循環 import になる（実測 ImportError）。
    ContextTask.__call__（=ワーカーのタスク実行時）まで生成を遅延することで、
    worker 起動対象インスタンスにも Flask app_context を提供する
    （2026-09-05 差分レビュー N1: app=None だと ContextTask が適用されず
    DB に触るタスクが無音失敗していた）。
    """
    global _flask_app
    if _flask_app is None:
        from app import create_app

        _flask_app = create_app()
    return _flask_app


celery = make_celery()


class ContextTask(celery.Task):
    """各タスク実行を Flask app_context 内で行う."""

    def __call__(self, *args, **kwargs):
        with _get_flask_app().app_context():
            return self.run(*args, **kwargs)


celery.Task = ContextTask
