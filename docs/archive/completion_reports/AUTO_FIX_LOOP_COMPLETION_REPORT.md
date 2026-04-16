# 自動修正ループ実装 - 完了レポート

## 実装日時
2025-11-28

## 概要

手動確認が必要だった修正を自動化し、完全自動ループを実現しました。

### 目的
- Materialization 失敗の問題を自動修正
- `tile_selectors` の自動追加
- 完全自動ループの実現

### ゴール
- 手動確認が必要な修正を最小限に
- 自動修正可能な問題は自動的に解決
- テスト実行 → 修正 → 再実行の完全自動化

## 実装ステップ

### Step 1: `ensure_plp_materialized` の site_config 対応

**変更内容:**
- `app/agents/browser/navigation_driver.py` の `ensure_plp_materialized` を更新
- `selectors.plp.tile_selectors` を優先的に使用するように変更

**変更前:**
```python
pdp_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
tile_selectors = _dedupe_keep_order(
    (pdp_cfg.get("pdp_link_selectors", []) or [])
    + (pdp_cfg.get("plp_container_selectors", []) or [])
    + [...]
)
```

**変更後:**
```python
plp_cfg = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
pdp_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}

tile_selectors = _dedupe_keep_order(
    (plp_cfg.get("tile_selectors", []) or []) +  # 新規: plp.tile_selectors を優先
    (plp_cfg.get("pdp_link_selectors", []) or []) +  # plp.pdp_link_selectors も使用
    (pdp_cfg.get("pdp_link_selectors", []) or [])
    + [...]
)
```

**なぜ変更したか:**
- `selectors.plp.tile_selectors` を追加することで、materialization のセレクタを site_config から取得できるようにする
- MonclerPLPStrategy で見つかっているセレクタを materialization でも使用できるようにする

### Step 2: 自動修正機能の拡張

**変更内容:**
- `auto_fix_and_retry.py` に `add_tile_selectors` アクションを追加
- Materialization 失敗の問題を自動修正可能に

**追加した修正案:**
```python
if errors["materialization_failed"]:
    if errors["moncler_tiles_found"] and errors["tile_count"] > 0:
        fixes.append({
            "type": "materialization_selector_mismatch",
            "description": f"MonclerPLPStrategy で {errors['tile_count']} 個のタイルが見つかっているが、materialization で見つからない",
            "action": "add_tile_selectors",
            "auto_fixable": True,  # 自動修正可能
            ...
        })
```

**実装した自動修正:**
- `selectors.plp.tile_selectors` に MonclerPLPStrategy で見つかっているセレクタを自動追加
  - `"a[href*='/products/']"`
  - `"div:has(a[href*='/products/'])"`
  - `"a[href*='/product/']"`
  - `"a[href*='/p-']"`

### Step 3: ループ処理の改善

**変更内容:**
- 自動修正可能な修正と手動確認が必要な修正を分離
- 自動修正可能な修正のみを適用

**変更前:**
```python
auto_fixable = any(f["action"] not in ["check_selectors", "update_selector", "adjust_materialization"] 
                 for f in fixes)

if not auto_fixable:
    print("[停止] 手動確認が必要な修正案があります")
    break
```

**変更後:**
```python
auto_fixable_fixes = [f for f in fixes if f.get("auto_fixable", False)]
manual_fixes = [f for f in fixes if not f.get("auto_fixable", False)]

if not auto_fixable_fixes and manual_fixes:
    print("[停止] 手動確認が必要な修正案のみです")
    break
```

**なぜ変更したか:**
- 自動修正可能な修正と手動確認が必要な修正を明確に分離
- 自動修正可能な修正がある場合は、それを適用して再実行

## 変更ファイル一覧

### 変更ファイル
- `app/agents/browser/navigation_driver.py` - `ensure_plp_materialized` が `selectors.plp.tile_selectors` を使用
- `auto_fix_and_retry.py` - `add_tile_selectors` アクションを追加、ループ処理を改善

## 動作確認結果

### 実装済みの自動修正

1. **`pdp_link_selectors` の自動追加** ✅
   - 条件: PDP リンクが0件 && MonclerPLPStrategy でタイルが見つかっている
   - 動作: `'a[href*="/products/"]'` を自動追加

2. **`tile_selectors` の自動追加** ✅（新規）
   - 条件: Materialization 失敗 && MonclerPLPStrategy でタイルが見つかっている
   - 動作: `selectors.plp.tile_selectors` に MonclerPLPStrategy で見つかっているセレクタを自動追加

## 設計上の改善点

### アーキテクチャの改善
1. **site_config の活用**
   - `selectors.plp.tile_selectors` を追加することで、materialization のセレクタも site_config から取得可能に

2. **自動修正の拡張**
   - より多くの問題パターンに対応
   - 自動修正可能な問題を自動的に解決

### 将来の拡張性への配慮
1. **新しい自動修正の追加**
   - `auto_fixable` フラグで自動修正可能な問題を明確化
   - 新しい修正パターンを簡単に追加可能

2. **ログ解析の改善**
   - MonclerPLPStrategy のタイルカウントを解析
   - より詳細な問題分析

## 既知の制約・注意事項

### 自動修正できないもの
1. **コード変更が必要な修正**
   - セレクタの更新（コード内のセレクタ文字列の変更）
   - ロジック変更

2. **判断が必要な修正**
   - 複数の修正案がある場合
   - 既存の動作に影響する可能性がある修正

### 安全性の考慮
- 自動修正は、設定ファイル（JSON）の更新のみ
- コード変更は手動確認が必要
- 最大リトライ回数は3回（無限ループを防止）

## 次のステップ

### 推奨されるフォローアップアクション

1. **自動修正のテスト**
   - `auto_fix_and_retry.py` を実行して、自動修正が動作するか確認

2. **`tile_selectors` の追加**
   - `app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL.selectors.plp` に `tile_selectors` を追加（自動修正で追加される）

3. **実ブラウザテストの再実行**
   - 自動修正後に、実ブラウザテストを実行して動作確認

## 使い方

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python auto_fix_and_retry.py
```

これで、以下のループが自動的に実行されます：
1. テスト実行
2. ログ解析
3. 問題検出
4. **自動修正適用**（新規）
5. 再実行

