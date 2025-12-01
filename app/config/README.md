# 設定ファイル構成

このディレクトリには、アプリケーションの設定ファイルが格納されています。

## ファイル構成

### `config.py`
**非機密設定のみ**を定義するクラス（`AppConfig`）。

- ブラウザ設定（HEADLESS、BROWSER等）
- ログ設定（LOG_LEVEL、LOG_FORMAT等）
- 環境設定（STAGE等）
- サイト設定（SITES）
- ディレクトリパス（ROOT_DIR、DATA_DIR等）

**機密情報（APIキー等）は含まれません。**

### `secrets.py` (自動生成)
**機密情報のみ**を格納するクラス（`Secrets`）。

- APIキー（GEMINI_API_KEY、DEEPSEEK_API_KEY、OPENAI_API_KEY等）
- データベースURL（DB_URL等）
- その他の機密情報

このファイルは `.env` から自動生成されます（`generate_secrets.py`を実行）。

**重要**: `secrets.py` は `.gitignore` に追加されており、Gitで追跡されません。

### `generate_secrets.py`
`.env` ファイルから `secrets.py` を自動生成するスクリプト。

**使用方法:**
```bash
python app/config/generate_secrets.py
```

### `config_loader.py`
後方互換性のための互換レイヤー。

- `Config.get(key, default)`: Secrets > .env > 環境変数の優先順位で値を取得
- サイト設定ローダ（`loader.py`）のAPIを委譲

### `loader.py`
サイト設定（`sites/base.json`等）を読み込むローダー。

## 設定の読み込み優先順位

### 非機密設定
1. 環境変数（`AK_BROWSER`、`AK_HEADLESS`等）
2. `AppConfig`クラスのデフォルト値

### 機密設定
1. `Secrets`クラス（`.env`から自動生成）
2. `.env`ファイル（直接読み込み）
3. 環境変数

## 使用例

### 非機密設定の取得
```python
from app.config.config import AppConfig

# ブラウザ設定
browser = AppConfig.BROWSER
headless = AppConfig.HEADLESS

# サイト設定
buyma_config = AppConfig.SITES.get("BUYMA")
```

### 機密設定の取得（推奨）
```python
from app.config.secrets import Secrets

# APIキー
gemini_key = Secrets.GEMINI_API_KEY
openai_key = Secrets.OPENAI_API_KEY
```

### 後方互換API（非推奨）
```python
from app.config.config_loader import Config

# 任意のキーを取得（Secrets > .env > 環境変数）
api_key = Config.get("GEMINI_API_KEY")
```

## .env ファイルの作成

1. プロジェクトルートに `.env` ファイルを作成
2. 必要な環境変数を設定:
   ```env
   GEMINI_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   DEEPSEEK_API_KEY=your_key_here
   DB_URL=sqlite:///./data/app.db
   AK_BROWSER=edge
   AK_HEADLESS=false
   ```
3. `generate_secrets.py` を実行して `secrets.py` を生成

## 注意事項

- **`.env` ファイルは Git で追跡しないでください**（`.gitignore` に追加済み）
- **`secrets.py` も Git で追跡されません**（`.gitignore` に追加済み）
- 機密情報は必ず `Secrets` クラスから取得してください
- 非機密設定は `AppConfig` クラスから取得してください

