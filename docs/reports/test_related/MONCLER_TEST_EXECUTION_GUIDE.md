# Moncler テスト実行ガイド

**作成日時**: 2025-11-30  
**目的**: Moncler テストを実行し、エラーメッセージを確認する

---

## 現在の状況

✅ **テストは実行されている**: ターミナル出力で「1 failed, 13 warnings」と表示  
❌ **エラーメッセージが不明**: テスト結果ファイルに詳細が記録されていない  
❓ **原因特定が困難**: WSL環境の制約でターミナル出力を直接確認できない

---

## 推奨される確認方法

### 方法1: WSL環境に直接ログインしてテストを実行（推奨）

1. **Windows PowerShell または Windows Terminal を開く**

2. **WSL Ubuntu にログイン**:
```bash
wsl
```

3. **プロジェクトディレクトリに移動**:
```bash
cd /home/yn441611/atelier-kyo-manager
```

4. **仮想環境を有効化**:
```bash
source venv/bin/activate
```

5. **テストを実行**（詳細なエラーメッセージを表示）:
```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

### 方法2: エラーメッセージをファイルに保存

1. **WSL環境にログイン**:
```bash
wsl
```

2. **プロジェクトディレクトリに移動**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
```

3. **エラーメッセージをファイルに保存**:
```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long > /tmp/moncler_test_error.log 2>&1
```

4. **エラーログを確認**:
```bash
cat /tmp/moncler_test_error.log
```

5. **Windows側からアクセス**:
```powershell
wsl cat /tmp/moncler_test_error.log
```

---

## 確認すべきポイント

### 1. テストが収集されているか

```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample --collect-only -v
```

このコマンドでテストが収集されるか確認してください。

### 2. テストの実行結果

```bash
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=short
```

このコマンドで、より簡潔なエラーメッセージが表示されます。

### 3. すべてのテストを実行

```bash
pytest tests/test_product_extractor.py -v
```

すべてのテストが実行されるか確認してください。

---

## 期待されるエラーメッセージの例

テストが失敗する場合、以下のようなエラーメッセージが表示される可能性があります：

### 例1: モックが正しく設定されていない
```
AttributeError: 'MagicMock' object has no attribute 'first'
```

### 例2: 価格正規化の問題
```
AssertionError: assert 1234.0 == 1234.56
```

### 例3: 設定取得の問題
```
KeyError: 'images'
```

---

## エラーメッセージを共有する方法

1. **エラーメッセージ全体をコピー**: ターミナルからエラーメッセージ全体をコピーしてください

2. **ファイルに保存**: エラーメッセージをファイルに保存して、共有してください

3. **スクリーンショット**: 必要に応じて、エラーメッセージのスクリーンショットを共有してください

---

## 修正済みの内容

以下の修正は既に完了しています：

1. ✅ `images` 設定の正規化（辞書形式対応）
2. ✅ Locator モックの2段階構造化（`.first` プロパティ対応）
3. ✅ `availability` 設定の正規化（辞書形式対応）

---

## 次のステップ

1. **エラーメッセージを確認**: 上記の方法でエラーメッセージを確認してください

2. **エラーメッセージを共有**: エラーメッセージを共有していただければ、原因を特定して修正します

3. **修正の適用**: エラーメッセージに基づいて、必要な修正を適用します

---

**ステータス**: 🔄 エラーメッセージ確認待ち

