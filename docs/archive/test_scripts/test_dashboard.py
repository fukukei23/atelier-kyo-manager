#!/usr/bin/env python3
"""
Dashboard起動テスト
"""
import sys
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')

print("=" * 70)
print("Dashboard起動テスト")
print("=" * 70)

# 1. アプリ作成テスト
print("\n1. Flaskアプリ作成テスト...")
try:
    from app.web import create_app
    app = create_app()
    print(f"   アプリ作成: OK")
    print(f"   アプリ名: {app.name}")
    print(f"   設定数: {len(app.config)}")
except Exception as e:
    print(f"   アプリ作成: 失敗")
    print(f"   エラー: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Blueprint登録確認
print("\n2. Blueprint登録確認...")
try:
    blueprints = list(app.blueprints.keys())
    print(f"   登録済みBlueprint: {blueprints}")
except Exception as e:
    print(f"   Blueprint確認: 失敗 - {e}")

# 3. URLルート確認
print("\n3. URLルート確認...")
try:
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    print(f"   ルート数: {len(rules)}")
    for rule in rules[:10]:
        print(f"   - {rule}")
except Exception as e:
    print(f"   ルート確認: 失敗 - {e}")

# 4. モデルインポート確認
print("\n4. モデルインポート確認...")
try:
    from app.models import Product
    print(f"   Productモデル: OK")
    print(f"   テーブル名: {Product.__tablename__}")
except Exception as e:
    print(f"   Productモデル: 失敗 - {e}")

# 5. 設定確認
print("\n5. 設定確認...")
try:
    print(f"   SECRET_KEY: {'設定済' if app.config.get('SECRET_KEY') else '未設定'}")
    print(f"   SQLALCHEMY_DATABASE_URI: {'設定済' if app.config.get('SQLALCHEMY_DATABASE_URI') else '未設定'}")
except Exception as e:
    print(f"   設定確認: 失敗 - {e}")

print("\n" + "=" * 70)
print("Dashboardテスト完了")
print("=" * 70)
