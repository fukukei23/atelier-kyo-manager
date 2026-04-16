# CR-ATELIER-002 Step 5-3: Locale / Trap / Search の扱いポリシー

- **Status:** Design Document
- **Author:** [AI Assistant]
- **Date:** 2025-12-08
- **Related CR:** CR-ATELIER-002_STEP5_MONCLER_ROBUST_EXTRACTION_DESIGN.md

## 1. 役割分担

### 1.1 LocaleGuard (`_ensure_expected_locale`)

**責務**:
- 現在のページ自体を `/en-int/...&shipToCountry=GB` に揃える
- 二重ロケールパターン（`/en-lt/en-int/...`）を検出して修正
- `goto` 後に再リダイレクトが発生した場合、再修正を試みる

**Pre-condition**:
- Moncler の PLP/検索 URL

**Post-condition**:
- `page.url` が `/en-int/...` で始まる
- 「明らかな Trap（検索トップ / ロケールゲート / 404）」でないこと

### 1.2 TrapDetector (`_detect_trap_page`)

**責務**:
- 明らかな Trap ページを検出
- 404 ページ、ロケールゲート、検索トップページを検出
- `TrapPageDetected` 例外を投げる

**検出パターン**:
- 404 ページ: `<h1>` に "It's not here" が含まれる
- ロケールゲート: 「Select your location」モーダルが表示されている
- 検索トップページ: 検索ボックスのみで、商品リストが存在しない

### 1.3 URL バリデーション (`_is_valid_moncler_pdp_url`)

**責務**:
- PDP 候補リンクをフィルタする
- 外部ドメイン、trap ページパターン、二重ロケールパターンを除外

## 2. Search ページの扱いポリシー

### 2.1 PLP 相当として扱う条件

**条件**:
- URL パターン: `/en-int/search` または `/en-lt/en-int/search`
- DOM 上に product tile が並んでいる
- `/products/` を含むリンクが一定数存在する（例: 5件以上）
- 明らかな検索トップページ（検索ボックスのみ）ではない

**実装方針**:
- `TrapDetector` で検索ページを検出した場合、DOM 構造を確認
- product tile が存在する場合は、PLP 相当として扱う
- ただし、ノイズの多い検索結果は除外する

### 2.2 Trap として扱う条件

**条件**:
- URL パターン: `/en-int/search` だが、商品リストが存在しない
- 検索ボックスのみが表示されている
- 404 ページ、ロケールゲート、二重ロケールパターンを含む

## 3. URL パターンの分類

### 3.1 許容 PLP URL パターン

- `/en-int/women/outerwear/all-down-jackets/`
- `/en-int/search`（product tile が存在する場合）
- `/en-int/.../products/...`（PDP URL）

### 3.2 Trap URL パターン

- `/en-int/404`
- `/en-int/not-found`
- `/en-int/client-service`
- `/en-lt/en-int/...`（二重ロケール）
- `/en-de/en-int/...`（二重ロケール）
- `/en-int/search`（product tile が存在しない場合）

## 4. リダイレクト挙動の防御策

### 4.1 Locale 補正後の再リダイレクト

**問題**:
- `/en-int/...` に補正したが、再び `/en-lt/en-int/...` にリダイレクトされる
- サーバ側のリダイレクトロジックによる可能性がある

**防御策**:
- `goto` 後に URL を再チェック
- 二重ロケールパターンが再発した場合、再修正を試みる（最大1回）

### 4.2 Search ページへのリダイレクト

**問題**:
- `/en-int/...` から `/en-lt/en-int/search` にリダイレクトされる
- ロケール不一致が原因の可能性がある

**防御策**:
- Search ページを検出した場合、DOM 構造を確認
- product tile が存在する場合は、PLP 相当として扱う
- 存在しない場合は、Trap として扱う

## 5. 実装方針

### 5.1 LocaleGuard の実装

1. **二重ロケールパターンの検出**:
   - `/en-lt/en-int/...` や `/en-de/en-int/...` を検出
   - 正規化して `/en-int/...` に修正

2. **再リダイレクトの検出**:
   - `goto` 後に URL を再チェック
   - 二重ロケールパターンが再発した場合、再修正を試みる

3. **Telemetry への記録**:
   - Locale 補正の回数を記録
   - 再リダイレクトが発生した場合、その情報を記録

### 5.2 TrapDetector の実装

1. **Search ページの検出**:
   - URL パターンが `/en-int/search` または `/en-lt/en-int/search` の場合
   - DOM 構造を確認して、product tile が存在するかチェック

2. **PLP 相当として扱う条件**:
   - product tile が存在する
   - `/products/` を含むリンクが一定数存在する

3. **Trap として扱う条件**:
   - product tile が存在しない
   - 検索ボックスのみが表示されている

### 5.3 URL バリデーションの実装

1. **Accept 条件**:
   - `origin == moncler.com`
   - `path == /en-int/.../products/...`
   - 二重ロケールパターンを含まない

2. **Reject 条件**:
   - 外部ドメイン
   - trap ページパターン（`/404`, `/not-found`, `/client-service`）
   - 二重ロケールパターン（`/en-lt/en-int/...`）

## 6. 次のステップ

1. **実装**:
   - LocaleGuard の再リダイレクト検出ロジックを実装
   - TrapDetector の Search ページ検出ロジックを実装
   - Telemetry への記録を実装

2. **テスト**:
   - 二重ロケールパターンの検出テスト
   - Search ページの PLP 相当判定テスト
   - 再リダイレクトの検出テスト

