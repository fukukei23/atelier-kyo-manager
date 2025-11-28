#!/bin/bash
# 完了レポートとチャット記録を docs/ 配下に移動するスクリプト

cd /home/yn441611/atelier-kyo-manager

# ディレクトリ作成
mkdir -p docs/completion_reports
mkdir -p docs/chat_records

echo "完了レポートを移動中..."

# 完了レポートを移動
find . -maxdepth 1 -name "*COMPLETION_REPORT*.md" -type f -exec mv {} docs/completion_reports/ \;

echo "チャット記録を移動中..."

# チャット記録を移動
find . -maxdepth 1 -name "CHAT_RECORD_*.md" -type f -exec mv {} docs/chat_records/ \;

echo "移動完了！"
echo ""
echo "完了レポート: docs/completion_reports/"
echo "チャット記録: docs/chat_records/"

