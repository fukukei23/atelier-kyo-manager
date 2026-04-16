# Stealth（指紋対策）共通化リファクタリング完了レポート

## 実装日時
2025年12月3日

## Reasoning（なぜこの変更を行ったか）

### Stealth を共通モジュールに切り出した意図

1. **コードの重複削減**: SessionManager と Moncler パッチで Stealth 関連のコードが重複していたため、共通モジュールに集約
2. **保守性の向上**: Bot 検知回避のロジックを一箇所に集約することで、将来的な改善や修正が容易になる
3. **再利用性の向上**: 他のサイトやエージェントからも Stealth 機能を簡単に利用できるようにする
4. **責務の明確化**: Stealth（汎用的な Bot 検知回避）とサイト固有の UI 操作（Moncler パッチ）の役割を明確に分離

### BrowserUseAgent / Moncler patch との責務分担

#### scraping/stealth.py（共通 Stealth モジュール）
- **責務**: すべてのサイトで共通の Bot 検知回避
  - User-Agent / Viewport の設定
  - `navigator.webdriver` の false 化
  - `navigator.permissions.query` のパッチ
  - Canvas / WebGL / Audio fingerprinting 対策
  - Locale / Timezone の設定

#### browser_use_moncler_patch.py（Moncler サイト固有パッチ）
- **責務**: Moncler サイト固有の UI 操作
  - Cookie 注入（`moncler-shipping-country`, `moncler-shipping-language`）
  - LocalStorage 設定
  - OneTrust バナーのクリック
  - ロケーションモーダルの処理
  - URL 正規化（`en-de` → `en-int` など）

#### SessionManager（セッション管理）
- **責務**: BrowserContext の作成と Stealth の適用
  - BrowserContext 作成時に Stealth パラメータを取得
  - `apply_stealth_to_context()` を呼び出して Stealth を適用

## Diff Summary（修正されたファイルと主要差分の要点）

### 新規作成ファイル

1. **scraping/stealth.py**
   - `build_stealth_params_from_site_config()`: site_config から Stealth パラメータを構築
   - `apply_stealth_to_context()`: BrowserContext に Stealth を適用
   - `navigator.webdriver` の false 化
   - `navigator.permissions.query` のパッチ
   - Canvas / WebGL / Audio fingerprinting 対策
   - Locale / Timezone の動的設定

### 変更ファイル

1. **app/agents/browser/session_manager.py**
   - `_build_context_options()`: Stealth パラメータを `build_stealth_params_from_site_config()` から取得するように変更
   - `_ensure_open()`: BrowserContext 作成後に `apply_stealth_to_context()` を呼び出すように変更
   - `_setup_init_scripts()`: Stealth モジュールが利用可能な場合は呼び出しをスキップ（Stealth モジュール内で処理）

2. **app/agents/browser_use_moncler_patch.py**
   - ファイルヘッダーに Stealth との役割分担を明記
   - バージョンを 2.2.0 に更新
   - Moncler サイト固有の UI 操作のみを担当することを明記

### 主要差分の要点

#### SessionManager の変更

**変更前:**
```python
context_opts = self._build_context_options()
context = await browser.new_context(**context_opts)
await self._setup_routes(context)
await self._setup_init_scripts(context)
```

**変更後:**
```python
context_opts = self._build_context_options()
context = await browser.new_context(**context_opts)

# Stealth を適用（context 作成後）
if apply_stealth_to_context is not None:
    stealth_params = build_stealth_params_from_site_config(...)
    await apply_stealth_to_context(context, **stealth_params)
else:
    # フォールバック: 既存の _setup_init_scripts を使用
    await self._setup_init_scripts(context)

await self._setup_routes(context)
# Stealth は既に適用済み
```

#### _build_context_options() の変更

**変更前:**
```python
def _build_context_options(self) -> Dict[str, Any]:
    ctx_opts: Dict[str, Any] = {}
    viewport = self.settings.get("viewport")
    if self.settings.get("enable_viewport_rotation"):
        viewport = random.choice(VIEWPORT_POOL)
    # ... 既存のロジック ...
```

**変更後:**
```python
def _build_context_options(self) -> Dict[str, Any]:
    ctx_opts: Dict[str, Any] = {}
    
    # Stealth パラメータを取得
    if build_stealth_params_from_site_config is not None:
        stealth_params = build_stealth_params_from_site_config(...)
        # context 作成時に設定可能な項目を設定
        if stealth_params.get("viewport"):
            ctx_opts["viewport"] = stealth_params["viewport"]
        # ... その他の設定 ...
    else:
        # フォールバック: 既存のロジックを使用
        # ...
```

## Next Action（次に行うべきこと）

### 1. 他サイト用の stealth プロファイル追加

- サイトごとに異なる Stealth 設定が必要な場合、`build_stealth_params_from_site_config()` にサイト固有のロジックを追加
- 例: Moncler 専用の UA を使う場合は `site_config` に設定を持たせ、Stealth がそれを読む

### 2. Moncler Phase1.5 (instance 再構築 + dry-run) で Stealth 導入後の実動確認

- Stealth 導入後の Moncler サイトでの動作確認
- Bot 検知回避が正しく機能しているか確認
- パフォーマンスへの影響を確認

### 3. テストの追加

- `scraping/stealth.py` のユニットテスト
- `build_stealth_params_from_site_config()` のテスト
- `apply_stealth_to_context()` のテスト
- SessionManager との統合テスト

### 4. ドキュメントの更新

- `scraping/stealth.py` の API ドキュメント
- 使用例の追加
- サイト固有の Stealth 設定方法のドキュメント

### 5. 既存コードの段階的な移行

- 他のエージェントやモジュールで直接 Stealth 関連のコードを書いている箇所を、`scraping/stealth.py` 経由に置き換え

## 動作確認結果

### 静的解析結果
- リンターエラー: なし
- 型チェッカー: 未実施（型ヒントは追加済み）

### コードレビュー結果
- Stealth モジュールと Moncler パッチの役割分担が明確
- SessionManager の変更は後方互換性を保つ（フォールバック機能あり）
- 既存の BrowserUseAgent の public API / 外部から見た挙動は変更なし

### テスト結果
- ユニットテスト: 未実施（今後のタスクとして推奨）
- 統合テスト: 未実施（今後のタスクとして推奨）

## 既知の制約・注意事項

### 既存コードとの互換性
- SessionManager は Stealth モジュールが存在しない場合、既存の `_setup_init_scripts()` を使用するフォールバック機能あり
- 既存の BrowserUseAgent の public API / 外部から見た挙動は変更なし

### 制限事項やトレードオフ
1. **BrowserContext の制約**: Playwright の BrowserContext は作成後に viewport / locale / timezone を直接変更できないため、init script で JavaScript 側から設定
2. **パフォーマンス**: Stealth スクリプトの注入により、ページ読み込み時間が若干増加する可能性がある
3. **サイト固有の設定**: サイトごとに異なる Stealth 設定が必要な場合は、`site_config` に設定を持たせる必要がある

### 移行時の注意点
- Stealth モジュールが存在しない場合、既存の `_setup_init_scripts()` が使用される（後方互換性あり）
- Moncler パッチは Stealth モジュールとは独立して動作（役割分担が明確）

## 関連ファイル

- `scraping/stealth.py`: Stealth 共通モジュール
- `app/agents/browser/session_manager.py`: SessionManager（BrowserContext 作成と Stealth 適用）
- `app/agents/browser_use_moncler_patch.py`: Moncler サイト固有パッチ
- `app/agents/browser_use_agent.py`: BrowserUseAgent（SessionManager 経由で Stealth を使用）

