# ======================================================================
# ファイル名: runserver.py
# 用途:
#   ローカル開発用のFlask起動スクリプト
#   "python runserver.py" でブラウザから確認できる
#
# 依存:
#   app/__init__.py の create_app()
# ======================================================================

from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug=True なのでホットリロードあり
    # host="0.0.0.0" は LAN からも見たいなら使ってもいい
    app.run(debug=True)
