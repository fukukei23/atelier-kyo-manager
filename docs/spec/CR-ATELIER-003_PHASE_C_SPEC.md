# CR-ATELIER-003 Phase C Spec  

BrowserUseAgent 分離・テストスイート再構築フェーズ  

**Version:** 1.0  

**Date:** 2025-12-09  

**Author:** System Architect (AI)

---

# 1. Overview & Context

Phase B にて、以下を達成した：

- Moncler 固有ロジックを BrowserUseAgent / NavigationDriver から完全分離し、  
  `app/agents/moncler/` 配下へ移動。
- BrowserUseAgent → ブランド非依存のオーケストレータ化が可能な状態になった。

しかし BrowserUseAgent は依然として **2700 行超の肥大化ファイル**であり、  
役割（UIオペレーション・ナビゲーション補助・オーケストレーション）が混在している。

また、既存テストの一部は old API を前提としており、  
現行アーキテクチャとは不一致のため **テスト失敗** が残っている。

Phase C は以下を目的とする：

1. **BrowserUseAgent をオーケストレーション層として分離整理する**
2. **テストスイートの更新・再構築**
3. **新たな単体テスト層の導入（モジュール分割後の品質保証）**

---

# 2. Goals / Acceptance Criteria

Phase C が完了したと判断する基準：

### 2.1 Architecture / Code
- BrowserUseAgent がオーケストレーション専用の軽量クラスとなっている  
 （実処理は NavigationDriver / ハンドラ / ヘルパーモジュールに移動済み）
- UI 操作系（クリック・入力）は `ui_helpers.py` に移動
- 低レイヤブラウザ操作は NavigationDriver の責務へ完全委譲

### 2.2 Tests
- 古い API を前提とする failing tests が修復されている  
  - `tests/test_browser_use_agent_plp_integration.py`  
  - `tests/test_navigation_driver_stage3a2.py`
- 新たに分離されたモジュール向けの単体テストが追加されている

### 2.3 Quality
- BrowserUseAgent のファイルサイズが **最低 25% 減少**  
- 依存関係が明確に整理された状態である  
- pytest が CI で完全グリーン

---

# 3. Scope

## 3.1 In-Scope（今回やること）

### **C-1: BrowserUseAgent の分離・再構成**
- BrowserUseAgent の責務整理（orchestrator に専念）
- 以下の新規ファイル（または既存ファイル）へ処理移動：

  | 新ファイル | 移動する責務 |
  |------------|--------------|
  | `browser_orchestrator.py` | PLP/PDP/自己修復フローの orchestration |
  | `ui_helpers.py` | ボタンクリック・要素存在確認など UI 操作 |
  | `navigation_helpers.py` | URL 正規化、redirect 判定、ナビゲーション補助 |
  | `browser_context_manager.py`（必要に応じて） | セッション・cookie 管理 |

- NavigationDriver との責務境界の最終確定

### **C-2: 既存テストの修復**
- 以下のテストを現行 API に合わせて更新：
  - `tests/test_browser_use_agent_plp_integration.py`
  - `tests/test_navigation_driver_stage3a2.py`

- 修正ポイント例：
  - coroutine-aware mock に差し替え (`AsyncMock`)
  - obsolete API 呼び出し（looks_like_trap_or_legal / ensure_plp_materialized）の削除・更新
  - NavigationDriver のコンストラクタ仕様に合わせた修正

### **C-3: 新規テスト追加（分離後の設計を保証）**
- Orchestrator テスト（BrowserUseAgent または新モジュール）
- UI ヘルパーテスト（要素検出／クリックの mock）
- NavigationHelper / URL 正規化テスト
- ブラウザ依存コードの mock 化規約整備（pytest fixtures 化）

---

## 3.2 Out-of-Scope（今回やらないこと）

- Moncler 抽出ロジックの追加改修（これは Step 4〜7 で完了）
- 新ブランド対応（別 CR）
- 実ブラウザの E2E 大規模検証（Phase D 以降）
- LLM（Self-Healing / Selector Discovery）精度改善

---

# 4. Architecture Plan（設計詳細）

## 4.1 BrowserUseAgent のターゲット構造（Phase C 後）

```
BrowserUseAgent (orchestrator)
├── browser_orchestrator.py（新規）
├── ui_helpers.py（UI操作系）
├── navigation_helpers.py（遷移・URL補助）
├── session_manager (既存)
└── NavigationDriver（低レイヤ）
```

### ★ BrowserUseAgent の役割は「フロー制御のみ」に限定する：

- Moncler / 他ブランドは handler に委譲  
- PLP 検索 → PDP 抽出 → Self-Healing → Telemetry という高レベルフローの構成だけを保つ  
- 実処理は NavigationDriver / handler / helper 側へ分解

---

# 5. Implementation Plan（実装計画）

## **C-1: BrowserUseAgent 分離**

### Step C-1-1: 分離対象の特定
- BrowserUseAgent から UI 操作（click, fill, wait）を抽出  
- URL 操作（redirect 判定・正規化）も抽出  
- PLP/PDP フローの orchestration 部分だけを残す

### Step C-1-2: ui_helpers.py の作成
- `click_element`
- `click_first_matching`
- `wait_for_selector_safe`
- `element_exists`

### Step C-1-3: navigation_helpers.py の作成
- `normalize_url`
- `is_redirect_loop`
- `is_wrong_locale`
- `ensure_locale`

### Step C-1-4: browser_orchestrator.py の作成
- `run_plp_flow`
- `run_pdp_flow`
- `handle_self_healing`
- BrowserUseAgent からこれらを完全移譲

### Step C-1-5: BrowserUseAgent のリファクタ
- フロー制御のみ残す（300〜600行程度を目安）
- 残った不要メソッドを削除

---

## **C-2: テスト修復**

### Step C-2-1: failing tests の調査
対象：
- test_browser_use_agent_plp_integration.py
- test_navigation_driver_stage3a2.py

### Step C-2-2: 修正
- AsyncMock 化
- obsolete API 削除
- constructor の signature に合わせる
- NavigationDriver へのモック差し替え

---

## **C-3: 新規テスト追加**

### Step C-3-1: Orchestrator テスト
- フロー制御が正しく handler / driver を呼ぶかを mock で検証

### Step C-3-2: ui_helpers テスト
- クリック成功/失敗の mock パターン
- 要素存在チェック

### Step C-3-3: navigation_helpers テスト
- locale 正規化
- redirect loop 判定
- URL の構文的正当性

---

# 6. Testing Strategy

### 6.1 pytest コマンド

```bash
python -m pytest tests -q -v
```

### 6.2 モジュール単位のテスト

```bash
python -m pytest tests/test_browser_orchestrator.py
python -m pytest tests/test_ui_helpers.py
python -m pytest tests/test_navigation_helpers.py
```

### 6.3 CI への統合
- Phase C 完了時点で GitHub Actions に pytest ワークフローを追加予定

---

# 7. Risks & Mitigation

| リスク | 内容 | 対策 |
|-------|------|------|
| BrowserUseAgent の呼び出し経路が複雑 | 既存ロジックが多岐に渡り、分離に時間がかかる | 分離順序を C-1-1 → C-1-5 の順に限定 |
| テストの老朽化 | 既存テストが API 変更についていけていない | Phase C の In-Scope にテスト修復を必ず含める |
| NavigationDriver との境界が不明確 | アンチパターン化していた歴史がある | helper 層の導入で明確化 |

---

# 8. Completion Criteria（完了条件）

- BrowserUseAgent の行数が **25%以上削減**されている  
 （例: 2,600行 → 1,900行前後）
- ui_helpers.py / navigation_helpers.py / browser_orchestrator.py が存在し、テストも通っている
- failing tests がすべて通る
- Moncler テスト 40件が引き続きグリーン
- 完了レポート（docs/completion_reports/CR_ATELIER_003_PHASE_C_COMPLETION_REPORT.md）が作成されている

---

# 9. Appendix: Phase C の開発順序（Cursor 作業ガイド）

1. BrowserUseAgent を開く  
2. 分離対象コードにコメントで目印をつける  
3. ui_helpers.py・navigation_helpers.py を新規作成  
4. ロジックを段階的に移動（削除しないで → 移植 → 呼び出し切り替え → 削除 の順）  
5. browser_orchestrator.py を作成しフロー制御を移行  
6. BrowserUseAgent の肥大メソッドを削除  
7. 既存テストが壊れるので C-2 で修復  
8. C-3 の追加テストを書く  
9. pytest / CI グリーン確認  
10. 完了レポートを作成  

---

# End of Spec

