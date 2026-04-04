# Atelier Manager 開発方針仕様書

**作成日**: 2026年3月21日
**更新日**: 2026年3月23日
**バージョン**: 1.1

---

## 1. プロジェクト概述

### 1.1 プロジェクトの目的
**Atelier Manager** は、concem ブランド向けの **EC商品管理与AI自動リサーチシステム**です。ブランド公式サイト（Moncler、SSENSE、Farfetchなど）から商品を自動探索・抽出し、利益が見込める商品を判定・レポートします。

### 1.2 技術スタック
| カテゴリ | 技術 |
|---------|------|
| Web FW | Flask + Flask-SQLAlchemy + Flask-Migrate + Flask-WTF |
| Browser | Playwright (async) + Selenium stealth |
| AI/ML | TensorFlow, PyTorch, rembg, ONNX, Google Generative AI, OpenAI |
| Data | pandas, BeautifulSoup4, icrawler |
| Testing | pytest |

---

## 2. 現状分析

### 2.1 主要コンポーネント现状

| コンポーネント | 版本 | 状態 | 備考 |
|-------------|------|------|------|
| BrowserUseAgent | v88.6.2J | 安定 | Moncler/SSENSE/GUCCI/Prada/FR対応済み |
| NavigationDriver | v2.0+ | 安定 | PLP materialize + Circuit Breaker統合 |
| MonclerPLPStrategy | v1 | 安定 | セレクタ・OneTrust対応完了 |
| FarfetchPlpStrategy | v1.0.0 | 追加済み | 2026-03-23追加 |
| SelfHealingAgent | v10.1.0J | 安定 | FKB統合 + Circuit Breaker |
| FKB (Failure Knowledge Base) | 22エントリ | 蓄積中 | `fkb_local.json` |
| AiResearchOrchestrator | v8.0.0J | 安定 | 最高司令部 |

### 2.2 完成済みタスク

| カテゴリ | タスク | 完了日 |
|---------|--------|--------|
| P0 | navigation_driver 改良 | 2026-03-21 |
| P0 | test_rembg.py 常時スキップ化 | 2026-03-21 |
| P1 | Moncler PLP 安定化 | 2026-03-21 |
| P1 | FKB構築 + Auto-Heal強化 (Circuit Breaker) | 2026-03-22 |
| P2 | Farfetch Plugin追加 | 2026-03-23 |
| UI/UX | ダッシュボードChart.js ID修正 | 2026-03-23 |
| UI/UX | index.html ダッシュボード導線追加 | 2026-03-23 |
| UI/UX | image_crawler.html 開発中バッジ | 2026-03-23 |
| UI/UX | list.html モバイル対応 | 2026-03-23 |
| UI/UX | Tailwind CDN固定化 (v3.4.1) | 2026-03-23 |
| UI/UX | Flashメッセージスタイル修正 | 2026-03-23 |
| UI/UX | CSVインポート/エクスポートUI強化 | 2026-03-23 |
| UI/UX | テーマ切り替え（ライト/ダーク） | 2026-03-23 |
| アーカイブ | base.legacy.html, form.html → docs/archive | 2026-03-23 |

### 2.3 既知の課題
1. **test_11.py** — Selenium GUIテストが элемент なしエラーで失敗（外部GUI依存）
2. **SSENSE Bot回避** — Cloudflare + WebGL fingerprinting 回避は現状困難と判明（優先度低）

---

## 3. 開発方針

### 3.1 基本原則
1. **小さなdiff**: 1回の操作で触るファイル数を最小限に
2. **拡張・オプション追加優先**: 破壊的変更よりも拡張を優先
3. **テスト重視**: テストを追加・修正してから本番コードを変更
4. **日本語コミュニケーション**: すべての応答・文档は日本語

### 3.2 残タスク

#### P0 - 今すぐ対応
| タスク | ファイル | 概要 |
|--------|---------|------|
| test_11.py 無効化/削除 | `tests/test_11.py` | Selenium GUI依存でCIで失敗、常時スキップまたは削除 |

#### P1 - 近期対応
| タスク | ファイル | 概要 |
|--------|---------|------|
| 収益性分析強化 | `app/agents/profitability_agent.py` | 利益計算精度向上 |
| SSENSE Bot回避研究 | - | Cloudflare対策（長期課題・優先度低） |

#### P2 - 中期対応
| タスク | ファイル | 概要 |
|--------|---------|------|
| バックアックファイル整理 | `app/routes_backup.py` 等 | archive移動・整理 |
| 収益性ダッシュボード强化 | `app/web/dashboard.py` | Chart.js機能拡張 |
| グローバル検索・フィルタ | - | 商品一覧の検索UI |

### 3.3 禁止事項
- `.env`, `instance/`, `logs/` などのユーザー固有ファイルを改変しない
- `git reset --hard`, `rm -rf` などの破壊コマンドを使用しない
- 本番コードの例外を握りつぶさない
- 条件分岐を「テストが通りやすいように」緩く変更しない

---

## 4. テスト戦略

### 4.1 テスト基本原则
- 1テスト = 1責務（1 assertion グループ）
- 内部実装詳細ではなく **公開APIの振る舞い** を検証
- 実際の LLM 呼び出しは禁止（すべてモック or スタブを使用）
- 外部 API / 時刻 / 乱数などに依存しないこと

### 4.2 テスト品質チェックリスト
- [ ] 関数名が `test_` で始まっているか
- [ ] Arrange / Act / Assert が明確に分離されているか
- [ ] 明示的な assert が最低1つあるか
- [ ] 成功ケース＋エラーケースを両方カバーしているか
- [ ] 実際の LLM 呼び出しを行っていないか
- [ ] 実行時間が 0.5 秒以内か

### 4.3 テスト実行方法
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true
python -m pytest tests/
```

---

## 5. 次のアクション

### 5.1 即座に実行
1. `test_11.py` を `@pytest.mark.skip` に変更または削除（Selenium GUI依存でCIで失敗するため）
2. バックアップファイル4件を `docs/archive/` へ移動
3. `profitability_agent.py` の強化余地を確認

### 5.2 結果レポート
テスト実行後、以下の內容を含むレポートを作成：
- 合計テスト数
- 成功/失敗/スキップ数
- 失敗したテストの一覧
- エラーの要約
- 修正の方向性

---

*この仕様書はプロジェクトの進行に応じて更新されます。*
