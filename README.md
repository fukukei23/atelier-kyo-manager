AtelierKyo Managerセットアップ手順このプロジェクトをローカル環境でセットアップし、実行するための手順です。1. 仮想環境のセットアップまず、プロジェクト用の独立したPython環境を作成し、有効化（アクティベート）します。Windows (コマンドプロンプト or PowerShell):python -m venv .venv
.\.venv\Scripts\activate
macOS / Linux (ターミナル):python -m venv .venv
source .venv/bin/activate
2. 依存ライブラリのインストールrequirements.txtファイルに記載されている、プロジェクトの実行に必要なライブラリをすべてインストールします。pip install -r requirements.txt
3. データベースの初期化次に、Flask-Migrateを使用してデータベースをセットアップします。Windows:set FLASK_APP=run.py
flask db init
flask db migrate -m "initial migration"
flask db upgrade
macOS / Linux:export FLASK_APP=run.py
flask db init
flask db migrate -m "initial migration"
flask db upgrade
注: flask db initコマンドは、プロジェクトで一番最初に一度だけ実行してください。2回目以降のデータベース更新ではflask db migrateとflask db upgradeのみを使用します。4. アプリケーションの実行以下のコマンドで開発サーバーを起動します。python run.py
サーバーが起動したら、Webブラウザで http://127.0.0.1:5000 にアクセスすると、アプリケーションのトップページが表示されます。