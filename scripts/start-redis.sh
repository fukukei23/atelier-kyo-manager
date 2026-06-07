#!/bin/bash
# Redisサーバーをバックグラウンド起動（WSL2用）
# 前提: sudo apt install redis-server
set -euo pipefail

if ! command -v redis-server &>/dev/null; then
    echo "❌ redis-server がインストールされていません"
    echo "   sudo apt install redis-server を実行してください"
    exit 1
fi

# 既に起動しているか確認
if redis-cli ping &>/dev/null; then
    echo "✅ Redis は既に起動しています (localhost:6379)"
    exit 0
fi

redis-server --daemonize yes --loglevel warning
sleep 0.5

if redis-cli ping &>/dev/null; then
    echo "✅ Redis 起動完了 (localhost:6379)"
else
    echo "❌ Redis 起動に失敗しました"
    exit 1
fi
