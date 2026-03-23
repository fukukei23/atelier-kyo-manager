# Codex セッションノート（再生成版・Moncler PLP 修復）

- **日時**: 2025-11-15  
- **プロジェクト**: `atelier-kyo-manager`  
- **実行環境**: Windows 11（従来環境、WSL なし）  
- **関連ファイル**:
  - `app/agents/browser_use_agent.py`
  - `app/agents/plugins/moncler_plp_v1.py`
  - `instance/runs/20251114_215425_998/run.json`
  - `instance/runs/20251114_215425_998/auto_heal_log.json`
  - `fkb_local.json`（Moncler PLP 失敗ナレッジ）

---

## 1. 失敗ケースの概要

- 実行例

  ```powershell
  (.venv) python run_orchestrator.py `
      --preset moncler_headful `
      --site MONCLER_OFFICIAL `
      --query "down jacket" `
      --auto-heal `
      --headful
  ```

- 期待: EN-INT (`https://www.moncler.com/en-int/women/outerwear/all-down-jackets/…`) で PLP が materialize し、product tiles を抽出できる。
- 実際: `run.json` に `PLP did not materialize (no product tiles)` と記録。`final_url` が EU/JP ロケールや `client-service/contact/` へ飛ぶケースあり。`dom_snapshot_path` の HTML には想定セレクタが存在せず、tile count が常に 0。

---

## 2. 原因の仮説

1. **PLP セレクタ不一致**  
   旧セレクタ `article[data-testid="product-card"]` に対し DOM が `div[data-testid="product-tile"]` 系へ変化。`count_tiles` が常に 0。

2. **ロケール・ゲート問題**  
   `monclergroup.com` ドメインや `/en-jp/`, `/client-service/contact/` へリダイレクト。ワープした際は PLP DOM が読み込まれず `likely_plp` 判定が false。

3. **Cookie / GDPR バナー**  
   OneTrust の `#onetrust-accept-btn-handler` による同意が必要。閉じないと PLP がロードされない。

4. **事前アクション不足**  
   ログインモーダル / 国選択モーダルを閉じる操作が足りず、`browser_use_agent` が product grid まで到達できない。

---

## 3. Codex 推定提案（修正要約）

### 3-1. PLP タイルセレクタ更新
- 対象: `site_config["selectors"]["plp"]` および `BrowserUseAgent` の抽出ロジック。
- 方針: 旧 `article[data-testid="product-card"]` を `div[data-testid="product-tile"]` や `a[data-test="product-card-link"]` 親要素へ更新。タイル数 < 10 の場合にバックアップセレクタを順番に試すフェイルオーバー。

### 3-2. OneTrust / モーダル対応
- `BrowserUseAgent.before_navigate` or `after_navigate` で `button#onetrust-accept-btn-handler` や `button[data-testid='close-modal']` をクリック。GDPR 同意なしで DOM が空になる問題を防ぐ。

### 3-3. ロケールトラップ検出
- `BrowserUseAgent._looks_like_trap_or_legal(url, dom)` に以下を追加:
  - `monclergroup.com` ドメイン
  - `/en-jp/`, `/brands/moncler`, `/client-service/contact`
- トラップ検出時は `logger.warning` を出し、`https://www.moncler.com/en-int/women/outerwear/all-down-jackets/` に強制遷移 → 再度 materialize を待つ。失敗した場合は FKB へ「ゲート突破失敗」として記録。

### 3-4. FKB 登録
- `fkb_local.json` へ以下のパターンを追加:
  - `error_signature`: `PLP did not materialize (no product tiles)` + `final_url` が `client-service/contact`.
  - `solution_pattern`: 「EN-INT PLP に戻す / OneTrust 同意 / セレクタ v2 を使う」といった復旧手順。
  - 目的: Auto-Heal が再発時にロジックを自動適用できるようにする。

---

## 4. 再実行ループ（人間オペ付き）

1. `moncler_plp_v1.py` と site config を修正 → `git diff` で確認。
2. Windows で headful 実行し、以下のサイクルを回す:
   - **Run #1**: 新セレクタの効果確認（tile カウント > 0 を目指す）
   - **Run #2**: トラップ検出/リカバリ検証（`looks_like_trap_or_legal` のログをチェック）
   - **Run #3**: FKB エントリが Auto-Heal で参照されるか確認
3. 成功 run の差分は `moncler_plp_v1.py` にコミット。失敗 run の diff は `codex_history/git_diffs` やドラフトブランチで保存。

---

## 5. 実行時の注意

- Playwright で外部サイトへアクセスする際はユーザー監視の下で実行。sandbox モードは静的解析・pytest 用にとどめる。
- `codex_history_manager` を使い、各 run 前後で `python3 tools/codex_history_manager.py` を実行して Codex Run Log と diff を自動保存。
- `.env`, `forward2me.json` などの機微情報はログに含めない。

---

## 6. 今後の改善アイデア

1. **Codex ログの定期バックアップ**  
   - `codex_history/` + `CodexHistoryManager` で `codex_chat_YYYYMMDD_HHMM.json` と `codex_diff_YYYYMMDD_HHMM.patch` を蓄積し、どの Run で何を試したか追跡。

2. **FKB × Codex の連携**  
   - `fkb_local.json` のエントリを Codex のコンテキストとしてロードし、類似エラー時に過去の「解決パターン」を即提案できるようにする。

3. **NexusCore/WSL 版への移植**  
   - Moncler PLP 対策を NexusCore 側の `nexuscore/agents/browser_use_agent.py` や FKB に反映し、Linux/WSL 実行でも自己修復ループを再現。

---

## 7. このノートの扱い

- **推定復元ノート**であり、元の Codex ノートを構造的に再現したもの。1 行単位で一致しているわけではないが、議論の骨子（Moncler PLP 失敗 → セレクタ更新 → トラップ検出 → FKB → 再実行）が揃っている。
- `.md` のまま保存しておくことで:
  - Codex へ「このノートを読んで続きから対応して」と指示できる。
  - NexusCore 側のエージェントに「Moncler 対策仕様」として読み込ませられる。
- 以後、`docs/codex_notes/` に日付別ノートを増やし、Codex セッションが失われても再構築できる状態をキープする。

---

### 今後の運用ステップ

1. `docs/codex_notes/` に本ファイルを置き、常にバージョン管理対象に含める。
2. 次回 Codex セッション開始時にこのノートを読み込ませ、「Moncler PLP 修復の続き」を指示する。
3. 新たな知見が得られたらノートを追記し、`codex_history` のログとも紐付ける。

