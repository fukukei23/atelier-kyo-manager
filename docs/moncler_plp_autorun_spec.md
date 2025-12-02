# Moncler PLP 自動回復フロー仕様書

- **作成日**: 2025-11-15  
- **対象**: `MONCLER_OFFICIAL` サイトにおける PLP スクレイピング  
- **目的**: Auto-Heal + Codex を組み合わせて「失敗 → LLM 提案 → 適用 → 再実行」を自律的に回すための運用仕様をまとめる。

---

## 1. 全体フロー

1. **実行**  
   - コマンド:  
     ```powershell
     cd C:\Users\USER\tools\atelier-kyo-manager
     .\.venv\Scripts\activate
     python run_orchestrator.py `
         --preset moncler_headful `
         --site MONCLER_OFFICIAL `
         --query "down jacket" `
         --auto-heal `
         --headful
     ```
   - 成功/失敗を問わず、Run Context が `instance/runs/<RUN_ID>/` に保存される。

2. **失敗時のアーティファクト**  
   - `fail_snapshot.md`, `failure_dom.html`, `screenshots/`, `videos/`, `auto_heal_log.json`。  
   - `auto_heal_log.json` には LLM 提案 (`selectors_patch`, `overrides_patch`, `url_rules`) が含まれる。

3. **提案の適用**  
   - コマンド:  
     ```powershell
     python tools/apply_llm_proposal.py --site MONCLER_OFFICIAL
     ```
   - `overrides.local.json` に差分が適用され、`git diff` で確認。

4. **再実行 & 記録**  
   - 上記 1〜3 を繰り返し、PLP materialize が成功するまでループ。  
   - Codex とのやり取りは必ず日本語で行い、ログは `docs/codex_notes/` に追記。  
   - `python3 tools/codex_history_manager.py --watch --interval 60` を別プロセスで常駐させ、Codex Run Log と Git diff を `codex_history/` に自動保存。

---

## 2. 主要コンポーネント

| 役割 | ファイル/モジュール | 主な機能 |
| ---- | ------------------ | -------- |
| CLI エントリ | `run_orchestrator.py` | 引数解析・プレイワーク起動・ログディレクトリ確保 |
| ブラウザ制御 | `app/agents/browser_use_agent.py` | Playwright セッション管理、Geo/Cookie 処理、PLP materialize、Moncler 専用ロジックのフック。v88.6.4 では URL 正規化ログと locale trap 再検知ログを追加。 |
| Moncler 戦略 | `app/agents/plugins/moncler_plp_v1.py` | URL 正規化、OneTrust 対応、`_PLP_TILE_SELECTORS` によるタイル検出（selector毎のカウントを RunContext log に出力）、ロケールモーダル操作 |
| サイト設定 | `app/config/sites/overrides.local.json` (`MONCLER_OFFICIAL`) | seed PLP URL、wait_for_selectors、selectors_patch、url_rules、pre_actions |
| Auto-Heal | `app/agents/gpt_integration.py`, `self_healing_agent.py` | 失敗ログ → LLM 提案生成 (`auto_heal_log.json`) |
| 提案適用 | `tools/apply_llm_proposal.py`, `tools/moncler_auto_cycle.py` | `apply_llm_proposal.py` がマージを担当。`moncler_auto_cycle.py` が run → proposal apply → (任意で git branch) を自動化。 |
| Codex 連携 | `docs/codex_instruction_manifest.md` | チャット開始時に読み込ませる指令書。会話は日本語で統一。 |
| ログ自動保存 | `tools/codex_history_manager.py` | `~/.codex/sessions` から Run Log/画像/diff を `codex_history/` へ同期 |

---

## 3. 運用ルール

1. **Command / Logging**  
   - 実行コマンドと結果は Codex チャットに必ず記録。  
   - `codex_history_manager.py --watch --interval 60` を起動し、Codex セッションログと `git diff` を随時バックアップ。

2. **メタ情報の記述**  
   - すべてのコード編集で、ファイル冒頭に「ファイル名 / レジストリ / 日付 / バージョン / 使用方法」をコメントで残す。  
   - スタブや仮実装には「スタブ」等の注記を入れる。

3. **日本語運用**  
   - Codex/Claude/GPT への指示・応答は日本語のみ。指令書 (`docs/codex_instruction_manifest.md`) を毎回読み込ませる。

4. **リスク管理**  
   - 自動適用された提案は `git diff` と `codex_history/git_diffs` で確認し、問題があれば revert。  
   - 冗長な run を避けるため、1 日数回の Cron 実行から開始し、挙動が安定してから常時監視を検討。

---

## 4. 監視・改善ポイント

1. **PLP タイル検知**  
   - `_PLP_TILE_SELECTORS` が正しくヒットしているか `browser_use_agent` ログにカウントを出力する。  
   - 0枚のままなら `selectors_patch.plp` を更新し、Auto-Heal 提案と差分を比較。

2. **ロケール正規化**  
   - `/en-xx/` ホームへの滞留を `_moncler_force_plp_if_locale_home` で検知し、ログに「Locale homepage detected」を出す。  
   - `run.json.message` に `/en-de/en-int` のような二重ロケールが出たら、`overrides.local.json.normalize_rules` を再調整。

3. **Auto-Heal 成果**  
   - `auto_heal_log.json` の `selectors_application.applied` が `true` になるか定期確認。  
   - `exports/llm_proposals/` を Git 管理し、どの提案が有効だったか追跡。
   - `_PLP_TILE_SELECTORS` のカウント結果（`[MonclerPLPStrategy] Tile counts ...`）でスクロール挙動を監視。

---

## 6. 自動化コンポーネント

1. **run → apply → commit ラッパー**  
   - `tools/moncler_auto_cycle.py` を使用。例:  
     ```bash
     python tools/moncler_auto_cycle.py \
         --site MONCLER_OFFICIAL \
         --query "down jacket" \
         --headful \
         --auto-heal \
         --apply-proposal \
         --git-branch moncler-auto-latest
     ```  
   - 実行フロー: run_orchestrator → 最新 Run の `auto_heal_log.json` を確認 → `llm_proposal_path` があれば `tools/apply_llm_proposal.py` を呼び出し → `--git-branch` が指定されていれば `git checkout -B <branch>` + `git add ...` + `git commit`。

2. **Codex 履歴同期**  
   - `python3 tools/codex_history_manager.py --watch --interval 60` を常時稼働。  
   - Run ごとに `codex_history/run_logs/*.jsonl`, `git_diffs/*.diff`, `metadata/*.json` を蓄積し、Run ID と紐付ける。

3. **ログ拡充**  
   - `browser_use_agent` で `/en-xx/en-yy/` の二重ロケールを検知すると `[_looks_like_trap] Detected double-locale path` を出力。  
   - `_normalize_to_en_int_url` と `_moncler_force_plp_if_locale_home` が URL の補正結果を `system.log` に書くため、後続の自動検証バッチが不整合を拾える。

---

## 7. リスクと対応

| リスク | 対応 |
| ------ | ---- |
| 過剰アクセス/サイト負荷 | Cron/Task Scheduler での実行間隔を 30〜60 分に設定し、検証中はヘッドレスモードを利用。 |
| 誤提案適用 | `moncler_auto_cycle.py` が提案適用時に `git diff` を出力。`codex_history/git_diffs` と Git branch で常に巻き戻せるようにする。 |
| 監視不足 | まずは `codex_history` や `instance/runs/**/run.json` をスキャンするシンプルな Python バッチを用意し、Slack/メール通知する。将来的には Grafana/Sentry などに転送。 |


---

## 5. スケジュール提案

| フェーズ | 目標 | 担当 |
|---------|------|------|
| Phase 1 | 1 日 2 回の自動実行 + Codex ログ保存を安定運用 | オペレーター |
| Phase 2 | Auto-Heal 提案の自動適用 → Git ブランチ生成をスクリプト化 | AI チーム |
| Phase 3 | 監視/通知（成功率、最終成功ラン ID）をダッシュボード化 | Infra/AI 共通 |
| Phase 4 | 完全無人化（Cron + 自動リリース + rollback ガード） | 組織承認後 |

**注意**: Phase 1-1 / Phase 1-2 の BrowserUseAgent レベルでの例外処理・retry ロジックの統一は完了しましたが、上位エージェント側（FailureAnalysisAgent / SelfHealingAgent / SelectorDiscoveryAgent）での例外分類情報の活用は、各エージェントのリファクタリングタスクとして別途実施予定です。詳細は `docs/official/refactoring/BROWSER_USE_AGENT_EXCEPTION_RETRY_REFACTOR.md` を参照してください。

---

この仕様書に従って運用すれば、Moncler PLP スクレイピングの失敗→自己修復ループを着実に回し、実運用レベルまで引き上げられる。
