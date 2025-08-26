# run.py (修正版)

from dotenv import load_dotenv

# アプリケーションをインポートする前に、.envファイルの内容を環境変数として読み込む
# これが最も重要です
load_dotenv()

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
