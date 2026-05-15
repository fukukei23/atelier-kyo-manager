---
name: コードレビュー指摘事項 — Opus × MiniMax (2026-05-15)
date: 2026-05-15
reviewer: Opus 4.7 / MiniMax M2.7
type: issue-backlog
status: open
---

# atelier-kyo-manager — レビュー指摘事項

> 出典: SSOT `40_CAREER/キャリア分析/01_能力評価/2026-05-15_クロスレビュー_Opus×MiniMax.md`
> 採用ポートフォリオ品質向上のための修正タスク。優先度順。

---

## 優先度: 🔴 高（採用面接で差別化に直結）

### ISSUE-001: ビジネスインパクトの数値化が README に無い
- **症状**: 「BUYMA×Buyandship 個人運用」と書いてあるが、月間取引件数・利益額・自動化により削減した工数 等の数値が一切ない
- **対象**: `README.md`
- **推奨対応**: README 冒頭に「Impact」セクション追加
  - 月間処理件数（注文数 / 出品数 / チャットボット応答数）
  - 自動化により削減した工数（時間/月）
  - LLM ルーティングによるコスト削減額（円/月）
  - チャットボットの一次対応率（自動応答率 % / エスカレ率 %）
- **想定工数**: 1〜2時間
- **完了条件**: 「個人事業の業務システムを実コードで回している」が一目で伝わる

### ISSUE-002: CI/CD パイプラインが README から見えない
- **症状**: `.github/workflows/` の有無・状態がリポジトリトップから不可視。バッジ無し
- **対象**: `.github/workflows/`（追加）, `README.md`
- **推奨対応**:
  - GitHub Actions で `ruff check + pytest -x --cov` を push/PR 時に走らせる
  - カバレッジ Codecov 連携
  - README に CI / Coverage バッジ追加
- **想定工数**: 半日
- **完了条件**: README 冒頭にバッジ 2 枚（CI + Coverage）

### ISSUE-003: テスト本体比が異常（テスト 13,924 行 / 本体 32,575 行 ≒ 0.4 倍だが、面接で問われる可能性）
- **症状**: テスト総 LOC は良いが、内容に over-mocking や冗長 fixture が混じっている可能性
- **対象**: `tests/*.py`（特に `test_browser_orchestrator_*`, `test_ai_*`）
- **推奨対応**:
  - 自己レビュー: 「実際の挙動を検証しているテスト」と「mock の戻り値を検証しているだけのテスト」を仕分け
  - 後者は削除 or integration test に昇格
  - README に「テスト戦略」セクションを 1 段落で書き、なぜこの行数なのかを説明できる状態にする
- **想定工数**: 1〜2日（自己レビュー）

---

## 優先度: 🟡 中（運用・スケール）

### ISSUE-004: SQLite + ローカルファイル前提でクラウドにそのまま乗らない
- **症状**:
  - DB が SQLite (`instance/` ローカル)
  - `diskcache` 500MB がローカルファイル（[`app/utils/ai_llm_controller.py:34-38`](app/utils/ai_llm_controller.py)）
  - Heroku / Cloud Run にそのままデプロイ不可
- **対象**: `app/config/config.py`, `app/utils/ai_llm_controller.py`
- **推奨対応**:
  - DATABASE_URL を環境変数化（既にされている場合は README に明記）
  - キャッシュ層を抽象化: `Cache` インターフェース → diskcache（local） / Redis（prod）
  - docker-compose.yml に Postgres + Redis を追加（ローカル開発でも疎通確認できる）
- **想定工数**: 2〜3日
- **完了条件**: `docker compose up` で全機能が動く

### ISSUE-005: ai_llm_controller がシングルトンで状態を持つためテスタビリティ低下
- **症状**: `AILlmController` がモジュールレベルシングルトン。テストで状態リセットが煩雑になる可能性
- **対象**: [`app/utils/ai_llm_controller.py:101-120`](app/utils/ai_llm_controller.py)
- **推奨対応**:
  - シングルトンは保ちつつ、`reset_instance()` クラスメソッドを公開（テスト専用）
  - Flask の `g` または DI コンテナ経由でリクエストスコープに移すことも検討（長期）
- **想定工数**: 半日

### ISSUE-006: モデルバージョン名がハードコード（`gemini-1.5-flash-latest`, `gpt-4o`, `deepseek-chat`）
- **症状**: モデル名 リテラルがコード内に直書き（[`app/utils/ai_llm_controller.py:51-53`](app/utils/ai_llm_controller.py)）。バージョン切替に毎回コード変更が必要
- **対象**: `app/utils/ai_llm_controller.py`, `app/config/`
- **推奨対応**:
  - `LLM_GEMINI_MODEL`, `LLM_OPENAI_MODEL` 等を環境変数化
  - デフォルト値はコード内に保持しつつ env で上書き可能に
- **想定工数**: 2〜3時間

---

## 優先度: 🟢 低（採用市場 + 体裁）

### ISSUE-007: ライセンスがコード上不明確（README は MIT 記載、LICENSE ファイル要確認）
- **対象**: `LICENSE`
- **推奨**: 既に MIT であれば問題なし。確認のみ

### ISSUE-008: requirements.txt 中の `# 最終更新: 2026-05-10` が古い
- **対象**: `requirements.txt`
- **推奨**: 依存更新時に自動でこの行を更新する pre-commit hook

### ISSUE-009: 英語 README 不在
- **対象**: `README.md`
- **推奨**: 冒頭の英語 abstract（既存の英語2行を拡張）

---

## 補足: 採用面接で語るべきストーリー

このリポジトリで強調すべきは:
1. **本番運用中の業務システム**（「自分の事業を実コードで回している」エンジニアの希少性）
2. **マルチLLMコントローラ**（[`app/utils/ai_llm_controller.py`](app/utils/ai_llm_controller.py): MODEL_COST テーブル、TASK_TO_MODEL_FAMILY、OpenTelemetry 計装）
3. **ステートマシン型自動発注**（[`app/services/auto_order_service.py`](app/services/auto_order_service.py): VALID_TRANSITIONS 宣言、AutoOrderLog 監査）
4. **18日ルールの業務ロジック**（決済方法別延長期限マッピング — ドメイン知識の完全コード化）

逆に質問されたら困る箇所:
- 「これは本番で動いてますか？スケールしますか？」 → ISSUE-001（数値）+ ISSUE-004（クラウド対応）を先に潰せ
- 「テストの数の根拠は？」 → ISSUE-003（テスト戦略の言語化）
