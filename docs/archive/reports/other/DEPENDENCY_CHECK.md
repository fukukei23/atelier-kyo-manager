# 依存関係チェック

**作成日時**: 2025-11-30

---

## 📋 requirements.txt の確認

`requirements.txt` には以下のテスト関連パッケージが含まれています：

- ✅ `pytest` - テストフレームワーク
- ✅ `pytest-asyncio` - 非同期テストサポート
- ✅ `playwright` - ブラウザ自動化

---

## 🔍 テストで使用されているパッケージ

### 標準ライブラリ（インストール不要）

- ✅ `unittest.mock` - Python標準ライブラリ（`AsyncMock`, `MagicMock`を含む）
- ✅ `typing` - Python標準ライブラリ

### 外部パッケージ

- ✅ `pytest` - `requirements.txt`に含まれている
- ✅ `playwright` - `requirements.txt`に含まれている
- ✅ `pytest-asyncio` - `requirements.txt`に含まれている

---

## ✅ 結論

**依存関係は問題ありません**。すべての必要なパッケージが`requirements.txt`に含まれています。

---

## 🔧 確認方法

依存関係を確認するには、以下のコマンドを実行してください：

```bash
# WSL環境で実行
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# 主要パッケージの確認
python -c "import pytest; print('pytest:', pytest.__version__)"
python -c "import playwright; print('playwright:', playwright.__version__)"
python -c "import pytest_asyncio; print('pytest-asyncio: OK')"

# すべての依存関係を再インストール（必要に応じて）
pip install -r requirements.txt
```

---

## 📦 インストールが必要な場合

仮想環境が壊れている場合や、新しい環境でセットアップする場合は：

```bash
# 仮想環境を作成（まだ存在しない場合）
python -m venv venv
source venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt

# Playwright ブラウザをインストール（初回のみ）
playwright install
```

---

**ステータス**: ✅ 依存関係は問題なし

