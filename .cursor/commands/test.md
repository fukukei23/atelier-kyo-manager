# /test : Atelier-Kyo テスト実行コマンド

あなたは Atelier-Kyo プロジェクトのテスト担当エージェントです。
ユーザーがこの `/test` コマンドを実行したら、次の手順で動いてください。

1. プロジェクトルートを確認し、必要なら説明だけ行う：
   - ルートは `/home/yn441611/atelier-kyo-manager` です。

2. 次のコマンドを提案し、実行する：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true
python -m pytest tests/
```

テスト結果を要約し、失敗テストがあれば修正方針を箇条書きで返す。

破壊的操作（`git reset --hard`, `git clean -fdx`, `rm -rf` 等）は一切提案しない。

---

これを入れると：

- 普段：**「テストして」** → 自然文トリガーでテスト
- 確実に：**`/test`** と打つと、必ずテスト実行フローに入る

という二段構えになります。
