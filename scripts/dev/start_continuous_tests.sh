#!/bin/bash
# テスト継続実行を開始するスクリプト

cd /home/yn441611/atelier-kyo-manager

# 仮想環境を有効化
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true

# バックグラウンドで実行
nohup python3 run_tests_continuous.py > /dev/null 2>&1 &

# プロセスIDを保存
echo $! > /tmp/run_tests_continuous.pid

echo "テスト継続実行を開始しました（PID: $(cat /tmp/run_tests_continuous.pid)）"
echo "ログファイル: test_continuous_log_*.txt"
echo ""
echo "確認コマンド:"
echo "  ps aux | grep run_tests_continuous"
echo "  tail -f test_continuous_log_*.txt"
echo ""
echo "停止コマンド:"
echo "  kill \$(cat /tmp/run_tests_continuous.pid)"

