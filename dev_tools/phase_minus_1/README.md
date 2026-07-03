# Phase -1 検証B: メルカリ安値滞留モニタ（使い捨て）

設計書 §7 検証B「先に買われる問題」を実測するための計測専用スクリプト。
**エンジン本体ではない。** 前提B（相場より安い出品が人間に買われる前に何分生き残るか）
を白黒つけるためだけに使い、判定が出たら破棄してよい。

## 何を測るか

- **発見**: 各キーワードを新着順検索し、`price_ceiling` 以下の出品を「安値候補」として登録。
- **SOLD確定**: 登録した候補の**個別商品ページ**を毎周ポーリングし、「売り切れました」表示で
  滞留時間（first_seen → sold）を確定。
  - 新着リストから消えた＝売れた、は使わない（新出品に押し出される交絡があるため）。

## 使い方

```bash
cd dev_tools/phase_minus_1
cp monitor_config.example.json monitor_config.json
# monitor_config.json の keyword と price_ceiling を編集
#   price_ceiling = オークフリー等で相場を調べ、手数料送料を引いても
#   利益が出る上限。相場より明確に安い出品だけを候補にする足切り。

../../venv/bin/python mercari_monitor.py --config monitor_config.json --headless
```

- `interval_sec`（既定900=15分）ごとに `rounds`（既定96=24時間）周回。
- 毎周 `out_dir`（既定 `results/`）に `sold_records.csv` と `summary.json` を上書き保存（中断耐性あり）。
- `Ctrl-C` で安全終了（その時点の結果を保存）。
- バックグラウンド継続例: `nohup ../../venv/bin/python mercari_monitor.py --config monitor_config.json --headless > monitor.log 2>&1 &`

## 出力

- `results/sold_records.csv`: 売れた候補ごとの `price / first_seen / sold_at / residence_min`。
- `results/summary.json`: `watched_total`（監視総数）/ `sold_total`（売却数）/
  `median_residence_min`（滞留中央値）/ `sold_within_10min`（10分以内に売れた数）。

## 判定（検証B通過ゲート）

- **RED**: 安値候補の多くが `residence_min` 数分（`sold_within_10min` が高比率）
  → 人間の巡回では間に合わない。自動監視＋即通知が必須、それでも厳しければ事業モデル見直し。
- **YELLOW**: 滞留は短いが十数分〜数十分ある → 通知速度の要件（何分以内に通知すれば買えるか）を
  この中央値から確定し、Phase 0 へ。
- **GREEN**: 十分な件数の安値が、現実的な巡回頻度で買える鮮度で存在 → Phase 0 へ。

## 注意

- `results/` はローカル計測データのため git 追跡しない（`.gitignore` 済み）。
- 対象サイトへの負荷に配慮し `interval_sec` は 900 秒以上を推奨。SOLD チェックは候補数に比例して
  時間がかかる（1件あたり数秒）。監視キーワードを増やしすぎない。
- セレクタ（`li[data-testid="item-cell"]` / `img alt` / 「売り切れました」）はメルカリの
  レイアウト変更で壊れうる。0件警告が続いたらセレクタを見直す。
