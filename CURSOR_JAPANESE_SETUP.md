# Cursor IDE 日本語化セットアップガイド

このプロジェクトを日本語環境で使用するための設定手順です。

## 手順1: 日本語言語パックのインストール

1. **拡張機能ビューを開く**
   - 左側のサイドバーで「拡張機能」アイコン（四角が4つ重なったマーク）をクリック
   - または `Ctrl + Shift + X`（Macの場合は `Command + Shift + X`）を押す

2. **日本語言語パックを検索・インストール**
   - 検索バーに「Japanese Language Pack for Visual Studio Code」と入力
   - Microsoft製の「Japanese Language Pack for Visual Studio Code」を探す
   - 「インストール」ボタンをクリック

## 手順2: 表示言語の設定

1. **コマンドパレットを開く**
   - `Ctrl + Shift + P`（Macの場合は `Command + Shift + P`）を押す

2. **言語設定を変更**
   - 「Configure Display Language」と入力して選択
   - リストから「日本語（ja）」を選択

3. **Cursorを再起動**
   - 「Restart Cursor to switch to 日本語？」というメッセージが表示されたら
   - 「Restart」をクリックしてCursorを再起動

## 設定ファイルについて

このプロジェクトには以下の設定ファイルが含まれています：

- `.vscode/locale.json` - プロジェクトの言語設定（日本語）
- `.vscode/settings.json` - プロジェクトレベルの設定
- `.vscode/extensions.json` - 推奨拡張機能の設定

これらのファイルにより、このプロジェクトを開いた際に自動的に日本語環境が推奨されます。

## トラブルシューティング

### 日本語化が反映されない場合

1. Cursorを完全に終了し、再度起動してください
2. 拡張機能が正しくインストールされているか確認してください
3. コマンドパレットから「Configure Display Language」を再度実行してください

### ユーザーレベルの設定を確認

プロジェクトレベルの設定が効かない場合は、ユーザーレベルの設定を確認してください：

- Windows: `%APPDATA%\Cursor\User\settings.json`
- Mac: `~/Library/Application Support/Cursor/User/settings.json`
- Linux: `~/.config/Cursor/User/settings.json`

以下の設定を追加してください：

```json
{
  "locale": "ja"
}
```

## 参考リンク

- [Japanese Language Pack for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=MS-CEINTL.vscode-language-pack-ja)

