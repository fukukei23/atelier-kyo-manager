# MONCLER PLP / PDP 抽出 凍結分析レポート

## 1. 目的
本ドキュメントは、MONCLER公式サイトにおける  
PLP → PDPリンク抽出処理が**設計前提ごと破壊されている**ことを記録し、  
今後の無駄な再調査・再実装を防ぐための凍結ログである。

---

## 2. 発生している現象（事実）

### 2.1 観測された挙動
- PLPにアクセス可能だが、以下の問題が恒常的に発生
  - product-card / product-tile DOMが安定しない
  - 初期DOMとmaterialized DOMで構造が変化
  - href抽出候補は存在するが、全reject（total_valid = 0）
- locale矯正後も `/en-jp/en-int/` の二重ロケールが継続
- click fallback / HAR fallback を含めてもPDP遷移に失敗

### 2.2 ログ・証跡
以下の成果物が生成され、再現性を確認済み：
- `plp_dom_initial_materialized.html`
- `selector_counts_plp_initial.json`
- `pdp_link_candidates_phase1.json`
- `pdp_link_candidates_phase2.json`
- `pdp_link_validation_report.json`
- `raw_pdp_links_v85.5.json`
- `network.har`
- `system.log`
- `trace.zip`

---

## 3. 原因分析（結論）

### 3.1 技術的結論
本問題は以下のいずれか、または複合要因によるものと判断する：

- Cloudflare / WAF による Bot 判定
- User-Agent / IP / 振る舞いによる段階的DOM変形
- React/Next.js による遅延描画＋A/B構造
- locale強制リダイレクトによる非決定的URL遷移

### 3.2 重要な判断
これは **バグでも実装不足でもない**。  
「HTML構造を前提にPLP→PDPを抽出する」という設計仮定自体が破壊されている。

---

## 4. 実施済み対策（やり切ったこと）

- URL正規化・判定ルール（CR-E2E-003B）実装
- locale安定化ロジックの冪等化
- trap判定例外処理
- HAR / Network fallback Phase 3 追加
- 全reject時の証跡保存保証
- 120s timeout時の診断情報保存保証

→ **これ以上のロジック修正で改善する見込みは低い**

---

## 5. 凍結判断

### 5.1 現時点でやらないこと
- セレクタ微調整の無限ループ
- click fallback の拡張
- 無条件なプロキシ導入

### 5.2 凍結理由
- 工数対効果が極端に悪い
- 他サイトでは同設計が正常動作する
- フレームワーク全体の価値を下げる

---

## 6. 再開条件（明示）

以下のいずれかを満たした場合のみ再検討する：

- 住宅IPプロキシ導入を前提とした検証を行うと決めた場合
- CDP / 実Chrome制御（非Playwright）に切り替える場合
- MONCLER側のPLP構造がHTMLベースに戻った場合
- 「Moncler専用ハードモード実装」を正式に切り出す場合

---

## 7. 現在の方針
MONCLERは **非安定サイト（Tier-D）** として扱い、  
E2E成功モデルの設計・実証対象から一旦除外する。

