#!/bin/bash
# Stage 3A-2 関連のカバレッジテストのみを実行

cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true

echo "=== Stage 3A-2 関連テストのカバレッジ測定 ==="
echo "対象ファイル:"
echo "  - app/agents/browser/navigation_driver.py"
echo "  - app/agents/browser_use_agent.py (NavigationDriver 統合部分)"
echo ""

# Stage 3A-2 関連のテストのみを実行
python -m pytest \
  tests/test_navigation_driver_stage3a2.py \
  tests/test_telemetry_service_stage3b.py \
  -v \
  --cov=app/agents/browser/navigation_driver \
  --cov=app/agents/browser_use_agent \
  --cov-report=term-missing \
  --cov-report=html:htmlcov/stage3a2 \
  --tb=short \
  -x  # 最初の失敗で停止

echo ""
echo "=== カバレッジレポート生成完了 ==="
echo "HTML レポート: htmlcov/stage3a2/index.html"

