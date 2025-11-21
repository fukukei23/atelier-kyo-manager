# ======================================================================
# ファイル名: tests/test_app_smoke.py
# 目的:
#   - Flaskアプリのファクトリ create_app() が例外なく動くか
#   - Blueprint が登録されているか
# ポイント:
#   - DBやLLMに依存せずに最低限の"起動できるアプリ"を保証する
# ======================================================================

import unittest
from app import create_app


class SmokeTestApp(unittest.TestCase):
    def test_app_factory_creates_flask_app(self):
        app = create_app()
        self.assertIsNotNone(app)
        # import_name は "app" のはず
        self.assertEqual(app.import_name, "app")

    def test_app_has_blueprint(self):
        app = create_app()
        # 実際のBlueprint名は routes.py の Blueprint("main", __name__)
        blueprints = [bp.name for bp in app.blueprints.values()]
        self.assertIn(
            "main",
            blueprints,
            msg=f"main blueprint が登録されていない: {blueprints}",
        )


if __name__ == "__main__":
    unittest.main()
