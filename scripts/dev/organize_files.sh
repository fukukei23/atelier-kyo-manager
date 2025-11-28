#!/bin/bash
# プロジェクトルートのファイルを整理するスクリプト

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# フォルダ作成
mkdir -p scripts/dev
mkdir -p docs/reports
mkdir -p tmp/generated
mkdir -p logs/generated
mkdir -p data/generated

echo "✅ フォルダを作成しました"

# 開発スクリプトを移動
echo ""
echo "📁 開発スクリプトを移動中..."
for file in check_*.py execute_*.py run_*_test*.py test_*.py verify_*.py quick_*.py show_*.py move_*.py organize_*.py measure_*.py fix_*.py remove_*.py auto_*.py debug_*.py final_*.py commit_and_push.py runserver.py run_orchestrator.py run_extractor.py run_browser_test.py run_all_tests.py multi_lang_crawler.py pricing_calculator.py buyma_research_tool.py 2>/dev/null; do
    if [ -f "$file" ]; then
        mv "$file" scripts/dev/ 2>/dev/null && echo "  ✅ $file -> scripts/dev/"
    fi
done

for file in run_coverage*.sh start_continuous*.sh verify_site_config*.sh check_browser_test.sh check_test_results.sh 2>/dev/null; do
    if [ -f "$file" ]; then
        mv "$file" scripts/dev/ 2>/dev/null && echo "  ✅ $file -> scripts/dev/"
    fi
done

# レポートを移動
echo ""
echo "📄 レポートを移動中..."
for file in *_REPORT.md *_ANALYSIS.md *_SUMMARY.md *_GUIDE.md *_RESULT.md *_STATUS.md *_VERIFICATION.md *_COMPLETION_REPORT.md *_TEST_RESULTS.md *_TEST_SUMMARY.md PROJECT_ANALYSIS_REPORT*.md STAGE_*.md AUTO_FIX_*.md BROWSER_TEST_*.md COVERAGE_*.md CURSOR_*.md FIX_*.md SITE_CONFIG_*.md SYSTEM_VALUATION_REPORT.md TEST_EXECUTION_*.md WSL_*.md CHAT_RECORD_*.md REFACTORING_PLAN.md PYTEST_USAGE.md TODO.md 2>/dev/null; do
    if [ -f "$file" ]; then
        mv "$file" docs/reports/ 2>/dev/null && echo "  ✅ $file -> docs/reports/"
    fi
done

# ログファイルを移動
echo ""
echo "📋 ログファイルを移動中..."
for file in *.log browser_test_*.log test_*.log test_output.log 2>/dev/null; do
    if [ -f "$file" ]; then
        mv "$file" logs/generated/ 2>/dev/null && echo "  ✅ $file -> logs/generated/"
    fi
done

# 一時ファイルを移動
echo ""
echo "🗑️  一時ファイルを移動中..."
for file in coverage.json depgraph.json project_dependency_graph.json git_status_check_result.txt tree.txt instance.zip page_source.html sample.html output.csv catalog_*.csv 2>/dev/null; do
    if [ -f "$file" ]; then
        mv "$file" tmp/generated/ 2>/dev/null && echo "  ✅ $file -> tmp/generated/"
    fi
done

# データファイルを移動
echo ""
echo "💾 データファイルを移動中..."
for file in catalog_data.csv catalog_images.csv 2>/dev/null; do
    if [ -f "$file" ]; then
        mv "$file" data/generated/ 2>/dev/null && echo "  ✅ $file -> data/generated/"
    fi
done

echo ""
echo "✅ ファイル整理が完了しました！"

