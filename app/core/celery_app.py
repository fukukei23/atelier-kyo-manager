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

    if app:

        class ContextTask(celery.Task):
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)

        celery.Task = ContextTask

    return celery


celery = make_celery()
