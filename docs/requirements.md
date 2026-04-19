# atelier-kyo-manager 機能要件

> **最終更新**: 2026-04-20
> **状態**: Phase 1 完了・Phase 2 計画中

---

## 1. システム概要

atelier-kyo-managerは、海外公式ブランドサイトから転送倉庫を経由してBUYMAへ出品する「無在庫転売モデル」に特化した業務管理・自動化システム。

利益最大化と業務効率化を実現するため、価格計算、在庫管理、画像処理、AIによる説明文生成などを統合的に管理する。当初は個人での週10-20時間の開発・運用制限を考慮し、月商30万円を達成するためのMVPを構築し、段階的に機能を拡張していく。

---

## 2. 機能要件

### Phase 1: コア機能（完了）

| ID | 機能名 | 優先度 | 状態 | 実装場所 |
|---|---|---|---|---|
| FR-001 | 簡易利益計算 | Must | ✅ 実装済み | `app/utils/pricing_calculator.py` |
| FR-002 | 画像AI処理（収集+背景除去） | Must | ✅ 実装済み | `app/services/image_service.py`, `app/utils/ai_image_crawler.py`, `app/utils/ai_background_remover.py` |
| FR-003 | AI説明文生成+出品テキスト | Must | ✅ 実装済み | `app/services/template_service.py`, `app/utils/ai_generate_descriptions.py` |
| FR-004 | BUYMA価格調査 | Must | 🔶 部分実装 | `app/services/price_scraper.py` (Issue #8) |
| FR-005 | 実ベース利益計算 | Must | ✅ 実装済み (Sprint 1) | `app/core/pricing/calculator.py` (13テスト) |
| FR-006 | 18日ルール管理 | Must | ✅ 実装済み (Sprint 2) | `app/models/order.py`, `app/routes/orders.py` (17テスト) |
| FR-007 | 出品候補リスト生成 | Must | ✅ 実装済み (Sprint 3) | `app/routes/products.py` + CSV出力 |

### Phase 2: 拡張機能（計画中）

| ID | 機能名 | 優先度 | 状態 | Issue |
|---|---|---|---|---|
| FR-008 | パイプライン一括実行・バックグラウンド化 | Should | 📋 計画中 | #15 |
| FR-009 | AI自動発注 & Slack通知 | Should | 📋 計画中 | #18 |
| FR-010 | AI ChatBot（顧客対応） | Should | 📋 計画中 | #19 |
| FR-011 | 出品完全自動化 | Should | 📋 計画中 | — |
| FR-012 | 発送通知API連携 | Should | 📋 計画中 | #17 |
| FR-017 | BUYMA出品CSV拡張 | Should | 📋 計画中 | #16 |
| FR-018 | Analyticsダッシュボード強化 | Could | 📋 計画中 | #20 |

### Phase 3: SaaS化（将来構想）

| ID | 機能名 | 優先度 | 状態 |
|---|---|---|---|
| FR-014 | マルチテナント・権限管理 | Won't | 未実装 |
| FR-015 | 海外公式サイト価格調査SaaS | Won't | 部分実装 |
| FR-016 | ChatBot自動海外仕入れ | Won't | 未実装 |

---

## 3. 非機能要件

| 項目 | 要件 | 状態 |
|------|------|------|
| 認証 | Flask-Login + パスワードハッシュ化 | ✅ 実装済み |
| 可用性 | ローカル/低価格クラウドで稼働 | ✅ |
| 性能 | 利益計算3秒以内、スクレイピングバックグラウンド | ✅ |
| 運用保守性 | モジュール疎結合、フォールバック可能 | ✅ |
| CI/CD | GitHub Actions自動テスト・デプロイ | 📋 Issue #21 |
| テストカバレッジ | 80%以上目標 | 📋 Issue #22 (現在231テスト) |
| コスト | 初期100万円以内 | ✅ |

---

## 4. 制約事項

### BUYMA規約遵守（違反 = アカウント停止）
- **禁止仕入れ先**: 国内公式オンライン、メルカリ等フリマアプリ
- **許可仕入れ先**: 海外オンライン（公式・セレクトショップ）、海外店舗、国内実店舗
- 18日以内発送（延長申請可能）
- 成約手数料 7.7%（国内ショッパー）

### 技術的制約
- スクレイピングはブロックリスク常時あり → 手動フォールバック必須
- 転送倉庫経由購入 → ブランド側の転売制限に抵触する可能性

### 開発制約
- ソロ開発、週10-20時間
- MVP優先、スモールスタート

---

## 5. 用語定義

| 用語 | 定義 |
|------|------|
| BUYMA | 海外ブランドアイテムの購入代行EC |
| 無在庫転売 | 在庫を抱えず、購入確定後に仕入れるモデル |
| 18日ルール | BUYMA購入者への発送完了を18日以内に行う規則 |
| pipeline_status | 商品のパイプライン処理状態 (pending/running/success/partial/failed) |
| 転送倉庫 | 海外購入品を日本へ転送するサービス (Buyandship等) |
| Bright Data | WebスクレイピングAPIサービス |
