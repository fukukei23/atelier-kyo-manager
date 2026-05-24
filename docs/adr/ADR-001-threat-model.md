# ADR-001: atelier-kyo-manager 脅威モデル

> ステータス: Accepted
> 日付: 2026-05-24
> 分類: Security

## コンテキスト

atelier-kyo-manager は BUYMA × Buyandship を利用した個人転売管理システム。Flask Webアプリとして以下の外部接続を持つ：

- 仕入先サイト（スクレイピング: requests/Playwright）
- BUYMA / Buyandship（ログイン自動化）
- 倉庫転送サービス Forward2me（Webhook受信）
- LLM API（Gemini/OpenAI/DeepSeek/MiniMax）
- Slack（通知）
- プロキシサービス（Webshare/BrightData）

### システム境界図

```
[仕入先サイト] ←スクレイピング─ [atelier-kyo-manager] ─Webhook受信→ [Forward2me]
[BUYMA] ←ログイン自動化─      │                                   [Slack通知]
[Buyandship] ←ログイン自動化─ │                                   [LLM API]
                              └─ SQLite/PostgreSQL（PII含む）
```

## 脅威一覧（STRIDE分類）

| ID | 脅威 | STRIDE | 影響 | 現状の対策 | ステータス |
|----|------|--------|------|-----------|-----------|
| T01 | 仕入先サイトの利用規約違反（スクレイピング） | Elevation | 法的リスク・IP ban | User-Agent偽装あり、rate limitなし | **未対応** |
| T02 |BUYMA/BuyandshipアカウントBAN（自動ログイン） | Elevation | ビジネス停止 | プロキシ経由、rate limitなし | **一部対応** |
| T03 | Webhook署名偽装 | Spoofing | 虚偽注文・荷物偽装 | HMAC-SHA256検証あり | **対応済** |
| T04 | .envファイルのシークレット露出 | Information Disclosure | 全サービス乗っ取り | .gitignoreに記載、暗号化なし | **一部対応** |
| T05 | 管理画面への不正アクセス | Spoofing | データ改ざん | Flask-Login + PBKDF2、MFAなし | **一部対応** |
| T06 | 顧客PIIの平文保存 | Information Disclosure | 情報漏洩 | 平文SQLite、暗号化なし | **未対応** |
| T07 | SQLインジェクション | Tampering | データ漏洩・改ざん | SQLAlchemy ORM主体、一部raw SQL | **低リスク** |
| T08 | LLMプロンプトインジェクション（チャットボット） | Elevation | 意図しない回答・コマンド実行 | テンプレートマッチ優先、サニタイズなし | **未対応** |
| T09 | プロキシサービス認証情報漏洩 | Information Disclosure | IP ban回避の無効化 | .env保存 | **一部対応** |
| T10 | レート制限なしによるDoS | Denial of Service | サービス停止 | 対策なし | **未対応** |

## 決定

### 優先対応（P0）

1. **T01 スクレイピングToS**: 対象サイトの利用規約を確認・記録。禁止されている場合は手動運用に切り替え。robots.txtの尊重を実装
2. **T04 シークレット管理**: `.env`をリポジトリから完全排除。`~/.secrets.env`に集約（既にSSOTで運用中）

### 推奨対応（P1）

3. **T05 認証強化**: ログイン試行回数制限（Flask-Limiter）の追加
4. **T08 チャットボット入力検証**: ユーザー入力のサニタイズ + LLM出力のフィルタリング
5. **T06 PII暗号化**: 顧客名・メールの暗号化保存（Fernet等）

### 受容（P2）

6. **T07 SQLi**: ORM主体のため低リスク。raw SQL使用箇所のパラメータ化クエリ確認のみ
7. **T10 DoS**: 個人利用のため受容。公開時に対応

## 結果

### 受容したリスク

- 個人利用システムのため、DoS（T10）とPII（T06）は当面受容
- スクレイピング（T01）は自己責任で継続、規約確認は必須

### 残タスク

- [ ] 対象サイト3件の利用規約・robots.txt確認（T01）
- [ ] .env → ~/.secrets.env 移行（T04）
- [ ] Flask-Limiter 導入（T05）
- [ ] チャットボット入力サニタイズ（T08）

## 参考

- STRIDE: Microsoft脅威モデリング分類（Spoofing/Tampering/Repudiation/Information Disclosure/Denial of Service/Elevation of Privilege）
- 対象リポジトリ: https://github.com/fukukei23/atelier-kyo-manager
