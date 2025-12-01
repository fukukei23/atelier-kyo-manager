# Moncler テスト修正サマリー

**修正日時**: 2025-11-30  
**問題**: `test_product_extractor_moncler_pdp_sample` が失敗（1 failed）  
**状況**: 複数回の修正を実施、テストが正常に実行されるように調整中

---

## 問題の原因

`_get_pdp_config()` が `images` 設定を辞書形式のまま返していたため、`_extract_images()` メソッドがリストを期待している部分でエラーが発生していました。

### テストコードの設定形式

```python
"images": {
    "selectors": [
        ".product-gallery img",
        ...
    ],
    "image_attr": "src",
    "base_url": None
}
```

### 期待される動作

`_extract_images()` は `pdp_config.get("images", [])` としてリストを期待しています。

---

## 修正内容

`app/agents/browser/product_extractor.py` に以下のヘルパーメソッドを追加：

### 1. `_normalize_images_config()` メソッド

```python
def _normalize_images_config(self, images_cfg: Any) -> Optional[List[str]]:
    """
    images 設定をリスト形式に正規化する
    
    Stage 5: 辞書形式（{"selectors": [...]}）とリスト形式の両方に対応
    """
    if not images_cfg:
        return None
    
    if isinstance(images_cfg, list):
        return images_cfg
    
    if isinstance(images_cfg, dict):
        return images_cfg.get("selectors")
    
    return None
```

### 2. `_get_image_attr()` メソッド

```python
def _get_image_attr(self, images_cfg: Any) -> Optional[str]:
    """images 設定から image_attr を取得"""
    if isinstance(images_cfg, dict):
        return images_cfg.get("image_attr")
    return None
```

### 3. `_get_image_base_url()` メソッド

```python
def _get_image_base_url(self, images_cfg: Any) -> Optional[str]:
    """images 設定から image_base_url を取得"""
    if isinstance(images_cfg, dict):
        return images_cfg.get("base_url")
    return None
```

### 4. `_get_pdp_config()` の修正

- `"images"` キー: `_normalize_images_config()` を使用してリスト形式に変換
- `"image_attr"` キー: 辞書形式の `images` からも取得できるように修正
- `"image_base_url"` キー: 辞書形式の `images` からも取得できるように修正

```python
"images": self._normalize_images_config(pdp_cfg.get("images")) or [
    ".product-images img",
    ...
],
"image_attr": self._get_image_attr(pdp_cfg.get("images")) or pdp_cfg.get("image_attr", "src"),
"image_base_url": self._get_image_base_url(pdp_cfg.get("images")) or pdp_cfg.get("image_base_url"),
```

---

## 修正後の動作

1. `images` が辞書形式の場合:
   - `selectors` キーからリストを取得
   - `image_attr` と `base_url` も辞書から取得

2. `images` がリスト形式の場合:
   - そのまま使用
   - `image_attr` と `base_url` は `pdp_cfg` から直接取得

3. `images` が存在しない場合:
   - デフォルトのリストを使用

---

## 次のステップ

テストを再実行して、修正が正しく動作するか確認してください：

```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -v
```

---

**ステータス**: ✅ 修正完了（テスト再実行待ち）

