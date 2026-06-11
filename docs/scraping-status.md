# ブランド公式サイト スクレイピング状況

> 最終更新: 2026-06-11（SSENSE全ブランドサングラス利益調査完了: 299品調査でSSENSE仕入れほぼ不可能と判明。Fendiのみ6品利益可能。代替仕入れ先Italist/Cettireを確定、Gucci Square Frame（Cettire $149）で利益率13.3%確認）
> **関連**: [経営者判断](経営者判断.md) — スクレイピング方針・EUR価格対応ブランドの最新状況
> 検証環境: WSL2 Ubuntu / curl_cffi (TLS偽装) + playwright-stealth (JS実行)

## ステータス定義

| アイコン | 意味 |
|---|---|
| ✅ | EU価格取得成功（BUYMA転売に利用可能） |
| ⚠️ | JP価格のみ取得可能（利益幅計算に不適） |
| ❌ | ブロック or 非公開 or 技術的に不可 |
| 🔒 | オンライン販売なし（店頭のみ） |

## 結果一覧

### ✅ 成功（EU価格取得）— スクレイパー統合済み

| ブランド | 国 | URL | 手法 | 通貨 | 商品数 | 備考 |
|---|---|---|---|---|---|---|
| **Gucci** | IT | `gucci.com/it/en/` | curl_cffi | EUR | 36 | aria-label正規表現 |
| **Prada** | IT | `prada.com/it/en/` | curl_cffi | EUR | 38 | aria-label正規表現 |
| **Valentino** | IT | `valentino.com/it-it/` | curl_cffi | EUR | 102 | data-canonical-url + data-price |
| **Versace** | IT | `versace.com/it/en/` | playwright-stealth | EUR | 46 | itemprop="price" |
| **Marni** | IT | `marni.com/it/en/` | playwright-stealth | EUR | 60 | itemprop="price" |
| **Chloe** | IT | `chloe.com/it/en/` | playwright-stealth | EUR | 10 | itemprop="price" |
| **Celine** | IT | `celine.com/it-it/` | playwright-stealth | EUR | 14 | class*="price" + EUR

### ⚠️ JP価格のみ

| ブランド | URL | 手法 | 通貨 | 商品数 | 備考 |
|---|---|---|---|---|---|
| **Ferragamo** | `ferragamo.com/shop/jpn/ja/` | playwright-stealth | JPY | 44 | EU版URL 404 |
| **Bottega Veneta** | `bottegaveneta.com/jp/ja/` | playwright-stealth | JPY | 8 | EU版は価格0 |
| **Balenciaga** | `balenciaga.com/jp/ja/` | curl_cffi | JPY | 28 | EU版 404（JP版は404でもHTML内に価格あり） |
| **Loewe** | `loewe.com/usa/en/` | curl_cffi | JPY | 165 | EU版→JPに強制リダイレクト |
| **Loro Piana** | `loropiana.com/jp/ja/` | curl_cffi | JPY | 8 | EU版HTTP/2エラー |

### ❌ 不可（EU版）

| ブランド | 試行URL | 手法 | 結果 | 原因 |
|---|---|---|---|---|
| **Saint Laurent** | `ysl.com/en-en/` | cffi+stealth | 価格null（425商品あり） | EU IP必要（サーバー側IP判定） |
| **Fendi** | `fendi.com/it/` | stealth | HTTP/2エラー | 強固なアンチボット |
| **Dolce & Gabbana** | `dolcegabbana.com/it-it/` | curl_cffi | 404→JP強制リダイレクト | IPベース地域制限 |
| **Givenchy** | `givenchy.com/it-it/` | stealth | Access Denied | アンチボット |
| **Burberry** | `burberry.com/` | stealth | ホームページにリダイレクト | IPベース地域制限 |
| **Tom Ford** | `tomford.com/` | curl_cffi | 404 | バッグページURL不明 |
| **Goyard** | `goyard.com/it/` | stealth | 404 | オンライン販売なしの可能性 |
| **Miu Miu** | `miumiu.com/` | stealth | 404/接続エラー | URL構造不明 |
| **Jacquemus** | `jacquemus.com/` | curl_cffi | 404 | URL構造不明 |
| **Brunello Cucinelli** | `brunellocucinelli.com/` | curl_cffi | 404 | URL構造不明 |
| **Jil Sander** | `jilsander.com/` | curl_cffi | 410 Gone | URL不明 |
| **The Row** | `therow.com/` | stealth | JPにリダイレクト | IPベース地域制限 |

### 🔒 オンライン販売なし（確認要）

| ブランド | 備考 |
|---|---|
| **Louis Vuitton** | オンライン購入不可（店頭のみ）。価格表示あり |
| **Hermès** | オンライン購入不可 |
| **Chanel** | オンライン購入不可 |

## 技術メモ

### 手法の使い分け

```
curl_cffi → 高速（数秒）・ブラウザ不要・TLS指紋偽装
  適用: Gucci(EUR), Prada(EUR), Valentino(EUR), Balenciaga(JP), Loewe(JP)

playwright-stealth → やや遅い（20秒）・ブラウザ起動・ボット回避
  適用: Ferragamo(JP), Bottega Veneta(JP), Versace(EUR), Marni(EUR), Chloe(EUR), Celine(EUR)
```

### EUR→JPY換算

`app/utils/fx_utils.py`のECB為替レートを自動使用。
現在のレート: €1 ≈ ¥184.5（2026-05-25時点）

### 「EU版URL不明」について

多くのブランドで`/it/en/`や`/it-it/`を試したが404。正しいEU版URLが存在する可能性あり。時間をかけてURLを特定すれば追加対応できるかも。

## 今後の改善案

1. **EU IP必要グループ**: YSL / D&G / The Row / Burberry / Loro Piana は商品データは取得できるが、価格がEU IPからでないとnull/リダイレクト。レジデンシャルプロキシ（月$10〜）で一括対応可能（5ブランド同時）
3. **セレクトショップ**: Mytheresa / NAP / 24S / LVR もレジデンシャルプロキシが必要
