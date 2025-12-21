# AKM Case Study Security Guidelines

**バージョン**: v1.0  
**作成日**: 2025-12-11  
**位置づけ**: AKM ケーススタディ領域のセキュリティ注意書き

---

## 1. 目的

本ドキュメントは、AKM ケーススタディ領域（`case_study/akm/`）における**機微情報の取り扱いルール**を定義する。

---

## 2. 基本方針

### 2.1 機微情報の禁止

**ケーススタディには機微情報を入れない。**

- API キー
- 認証トークン
- Cookie
- 個人情報
- 本番環境のURL（クエリ付き）

### 2.2 サンプルの扱い

**サンプルは必ず REDACTED / ダミー化する。**

- 実際の値を記載しない
- `REDACTED` または `***` で置き換える
- ダミー値を使用する場合は「ダミー値」と明記

### 2.3 .env.template の扱い

※ 注意：
`.env.template` が Git 履歴に存在するのは仕様であり問題ではない。
危険なのは実際の秘密情報を含む `.env` 本体のみである。

---

## 3. URL / ログの最小化ルール

### 3.1 URLの扱い

- **クエリ付きURLは載せない**
  - NG: `https://example.com?token=abc123&key=xyz`
  - OK: `https://example.com` または `https://example.com?token=***`

- **ベースURLのみ記載**
  - 例: `https://www.moncler.com`（クエリパラメータなし）

### 3.2 ログの扱い

- **ログは最小化**
  - 必要最小限の情報のみ
  - 認証情報を含むログは除外

- **サンプルログはダミー化**
  ```json
  {
    "api_key": "REDACTED",
    "user_id": "dummy_user_123",
    "timestamp": "2025-01-01T00:00:00Z"
  }
  ```

---

## 4. サンプルコードの扱い

### 4.1 環境変数の使用

```python
# NG: 実際のAPIキーを記載
api_key = "sk-1234567890abcdef"

# OK: 環境変数参照
api_key = os.getenv("OPENAI_API_KEY")

# OK: ダミー値（明記）
api_key = "REDACTED"  # ダミー値
```

### 4.2 設定ファイルの扱い

```json
{
  "api_key": "REDACTED",
  "database_url": "postgresql://user:***@localhost/db",
  "secret": "***"
}
```

---

## 5. ドキュメントの扱い

### 5.1 スクリーンショット

- 認証情報が含まれる画面は除外
- 必要に応じてマスク処理

### 5.2 コード例

- 実際のキー・トークンを含めない
- 環境変数参照または `REDACTED` を使用

---

## 6. チェックリスト

ケーススタディを追加・更新する際は、以下を確認する。

- [ ] API キー・認証情報が含まれていないか
- [ ] URL にクエリパラメータ（認証情報）が含まれていないか
- [ ] ログに機微情報が含まれていないか
- [ ] サンプルコードに実際の値を記載していないか
- [ ] ダミー値を使用している場合は「ダミー値」と明記しているか

---

## 7. 関連ドキュメント

- `docs/framework/SECURITY_BASELINE.md`: AASDF セキュリティベースライン
- `docs/framework/AASDF_V1_FREEZE_DECLARATION.md`: AASDF 凍結宣言

---

**END OF DOCUMENT**

