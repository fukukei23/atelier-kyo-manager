---
title: 注文管理
parent: 機能ショーケース
nav_order: 3
---

# 注文管理

![注文管理]({{ site.baseurl }}/screenshots/03-orders.png)

## 概要

注文の状態遷移（pending → sourcing → shipped → completed）を管理。18日ルールに基づく延長期限を決済方法別に自動計算します。

## 18日ルールとは

BUYMAでは商品到着後18日以内にバイヤーに発送する必要がある。 결제手段（クレジットカード/銀行振込）によって延長期限が異なる。

| 決済方法 | 延長期限 | BUYMA手数料 |
|---|---|---|
| クレジットカード | 45日 |
| 銀行振込 | 90日 |
| キャリア決済 | 30日 |

## 自動発注ステートマシン

```
pending ──→ sourcing ──→ cart_added ──→ checkout ──→ payment_done ──→ shipped ──→ completed
  │            │             │              │             │              │            │
  └─期限超過───┴─────────────┴──────────────┴─────────────┴──────────────┴────────────┘
                              (自動エスカレーション通知)
```

### 状態の詳細

| 状態 | 説明 | 自動アクション |
|---|---|---|
| pending | 新規注文受付 | Slack通知・自動発注開始 |
| sourcing | 仕入先を探している最中 | Buyandship価格検索 |
| cart_added | 商品をカートに入れた | 決済待機タイマー開始 |
| checkout | 決済完了 | 倉庫に発送指示 |
| payment_done | 入金確認済み | 追跡番号取得待ち |
| shipped | 発送済み | バイヤーへ追跡番号通知 |
| completed | 取引完了 | 利益計算・実績記録 |

## 技術的詳細

- **ステートマシン**: `auto_order_service.py` がPython状態機械で状態遷移を管理
- **延長期限計算**: `order.py` モデルのビジネスロジックで決済方法別に日付計算
- **Slack通知**: `notification_service.py` で注文イベントをSlackにリアルタイム通知
