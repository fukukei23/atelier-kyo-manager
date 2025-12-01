# Moncler 用 site_config チューニング - Step 1: 現状分析

**作成日時**: 2025-01-28  
**目的**: Moncler の site_config を Stage 4/5 新スキーマに合わせて分析・整理

---

## Step 1: 現状 Moncler の site_config を特定・把握

### 1. 設定ファイルの場所

- **メイン設定**: `app/config/sites/overrides.local.json` (88行目〜664行目)
- **キー名**: `MONCLER_OFFICIAL`

---

## 2. Moncler 設定ブロックの分類

### ✅ PLP 関連（PLP Driver 用）- Stage 4 対応済み

#### `selectors.plp.*` (290-327行目) - ✅ **新スキーマ化済み**

```jsonc
"selectors": {
  "plp": {
    "container_selectors": [...],      // ✅ Stage 4 新スキーマ
    "card_selectors": [...],           // ✅ Stage 4 新スキーマ
    "pdp_link_selectors": [...],       // ✅ Stage 4 新スキーマ
    "price_selectors": [...],          // ✅ Stage 4 新スキーマ
    "visible_price_selectors": [...],  // ✅ Stage 4 新スキーマ
    "tile_selectors": [...]            // ✅ Stage 4 新スキーマ
  }
}
```

#### `navigation.plp.*` (137-145行目) - ✅ **新スキーマ化済み**

```jsonc
"navigation": {
  "plp": {
    "supports_header_search": true,
    "supports_card_click_fallback": true,
    "supports_locale_normalization": true,
    "min_pdp_links": 12,
    "max_scroll_iterations": 12,
    "scroll_pause_ms": 350,
    "plp_timeout_ms": 35000
  }
}
```

#### `navigation.overlays.*` (111-126行目) - ✅ **新スキーマ化済み**

```jsonc
"navigation": {
  "overlays": {
    "cookie_banner_selectors": [...],     // ✅ Stage 4 新スキーマ
    "geo_modal_selectors": [...],         // ✅ Stage 4 新スキーマ
    "generic_close_buttons": [...]        // ✅ Stage 4 新スキーマ
  }
}
```

#### `navigation.trap.*` (92-100行目) - ✅ **新スキーマ化済み**

```jsonc
"navigation": {
  "trap_url_patterns": [...],      // ✅ Stage 4 新スキーマ
  "legal_url_patterns": [...]      // ✅ Stage 4 新スキーマ
}
```

---

### ⚠️ PDP 関連（ProductExtractor 用）- Stage 5 部分対応

#### `selectors.pdp.*` (356-468行目) - ⚠️ **旧スキーマ混在**

**現状の問題点**:

1. **旧スキーマキーが残存**:
   - `plp_container_selectors` (357-369行目) - ❌ これは PLP 用（削除対象）
   - `pdp_link_selectors` (370-386行目) - ❌ これは PLP 用（削除対象）
   - `plp_verification_selectors` (387-391行目) - ❌ これは PLP 用（削除対象）
   - `pdp_click_fallback_selectors` (392-400行目) - ❌ これは PLP 用（削除対象）

2. **PDP 抽出用のセレクタは存在するが、Stage 5 新スキーマに完全移行していない**:
   - `title_selectors` (401-406行目) - ⚠️ 旧キー名（`selectors.pdp.title` に統合すべき）
   - `price_selectors` (407-414行目) - ⚠️ 旧キー名（`selectors.pdp.price` に統合すべき）

3. **Stage 5 新スキーマで必要なキーが不足**:
   - ❌ `selectors.pdp.images` - 未定義
   - ❌ `selectors.pdp.colors` - 未定義
   - ❌ `selectors.pdp.sizes` - サイズ選択用はあるが、抽出用は未定義
   - ❌ `selectors.pdp.description` - 未定義
   - ❌ `selectors.pdp.breadcrumbs` - 未定義
   - ❌ `selectors.pdp.sku` - 未定義
   - ❌ `selectors.pdp.availability` - 未定義
   - ❌ `selectors.pdp.json_ld` - JSON-LD フォールバック設定が未定義
   - ❌ `selectors.pdp.image_attr` - 画像属性指定が未定義
   - ❌ `selectors.pdp.image_base_url` - 画像ベース URL が未定義
   - ❌ `selectors.pdp.raw_html_capture` - HTML 保存設定が未定義

4. **価格正規化ルールが未定義**:
   - ❌ `price_rules` または `selectors.pdp.price.normalize_rules` - 未定義

5. **サイズ選択関連はあるが、Stage 5 形式に合わせる必要**:
   - `price_requires_size` (443行目) - ⚠️ これは残す（サイズ選択が必要かどうか）
   - `auto_select_size` (444行目) - ⚠️ これは残す（自動サイズ選択するかどうか）
   - `size_container_selectors` (445-449行目) - ✅ これは残す（Stage 5 では `size_button` として統合可能）
   - `size_option_selectors` (450-455行目) - ✅ これは残す（Stage 5 では `size_button` として統合可能）
   - `size_selected_check_selectors` (456-460行目) - ✅ これは残す
   - `price_after_size_wait_selectors` (461-467行目) - ✅ これは `visible_price_selectors` として統合可能

---

### ⚠️ Trap / Overlay / Navigation 関連 - Stage 4 対応済み

- ✅ `navigation.trap_url_patterns` - Stage 4 新スキーマ
- ✅ `navigation.legal_url_patterns` - Stage 4 新スキーマ
- ✅ `navigation.overlays.*` - Stage 4 新スキーマ

---

### ❌ 旧スキーマ（削除・統合対象）

#### `selectors_patch.*` (528-584行目) - ❌ **旧スキーマ、削除対象**

- `selectors_patch.plp.*` - Stage 4 の `selectors.plp.*` に統合済み
- `selectors_patch.pdp.*` - Stage 5 の `selectors.pdp.*` に統合すべき
- `selectors_patch.modals.*` - Stage 4 の `navigation.overlays.*` に統合済み

#### `overrides_patch.*` (585-643行目) - ❌ **旧スキーマ、削除対象**

- Stage 4/5 の新スキーマに統合済み（ただし、一部は残す必要がある可能性）

---

## 3. Stage 5 スキーマとの対応状況

### ✅ 完全に新スキーマ化済み

- **PLP 関連**: `selectors.plp.*`, `navigation.plp.*` - Stage 4 完了
- **Overlay / Trap**: `navigation.overlays.*`, `navigation.trap_url_patterns` - Stage 4 完了

### ⚠️ 旧キーが残っている

- **PDP 抽出用セレクタ**: `selectors.pdp.title_selectors`, `selectors.pdp.price_selectors` など
  - これらは `selectors.pdp.title`, `selectors.pdp.price` に統合すべき

### ❌ まだハードコード依存している / 未定義

1. **PDP 抽出用セレクタの不足**:
   - `selectors.pdp.images` - 未定義
   - `selectors.pdp.colors` - 未定義
   - `selectors.pdp.sizes` - 抽出用が未定義（サイズ選択用はある）
   - `selectors.pdp.description` - 未定義
   - `selectors.pdp.breadcrumbs` - 未定義
   - `selectors.pdp.sku` - 未定義
   - `selectors.pdp.availability` - 未定義

2. **価格正規化ルール**: `price_rules` または `selectors.pdp.price.normalize_rules` - 未定義

3. **JSON-LD フォールバック**: `selectors.pdp.json_ld.*` - 未定義（ただし `selectors_patch.pdp.price_alt` に JSON-LD パスが含まれている）

4. **画像関連設定**: `selectors.pdp.image_attr`, `selectors.pdp.image_base_url` - 未定義

5. **HTML 保存設定**: `selectors.pdp.raw_html_capture` - 未定義

---

## 4. 次のステップ

### Step 2 で実施すること

1. Moncler 実 HTML / fixture を使ったギャップ分析
   - `instance/runs/*/failure_dom.html` をサンプルとして解析
   - 実際の DOM 構造に基づいて「理想的な selectors.pdp.*」を書き出す

2. 現在の site_config との突き合わせ
   - 「拾えている箇所」vs「拾えていない箇所」を特定

### Step 3 で実施すること

1. Stage 5 スキーマに準拠した JSON スニペットを作成
2. 差分パッチ（追加／上書き／削除）を提示

---

## 5. 現状のまとめ

### ✅ 良い点

- PLP 関連は Stage 4 で完全に新スキーマ化済み
- Overlay / Trap 処理も Stage 4 で新スキーマ化済み

### ⚠️ 改善が必要な点

- PDP 抽出用セレクタが旧スキーマのまま
- Stage 5 で必要な多くのキーが未定義
- 価格正規化ルールが未定義
- JSON-LD フォールバック設定が未定義

### ❌ 削除すべき点

- `selectors_patch.*` - 旧スキーマ、削除対象
- `overrides_patch.*` - 旧スキーマ、削除対象
- `selectors.pdp.plp_container_selectors` - PLP 用なので PDP から削除
- `selectors.pdp.pdp_link_selectors` - PLP 用なので PDP から削除

---

**次**: Step 2 で Moncler 実 HTML を解析して、具体的なセレクタを提案します。

