# Atelier Manager 開発方針仕様書

**作成日**: 2026年3月21日
**バージョン**: 1.0

---

## 1. プロジェクト概述

### 1.1 プロジェクトの月的
**Atelier Manager** は、concem ブランド向けの **EC商品管理与AI自動リサーチシステム**です。品牌公式サイト（Moncler、SSENSEなど）から商品を自動探索・抽出し、利益が見込める商品を判定・レポートします。

### 1.2 技術スタック
| カテゴリ | 技術 |
|---------|------|
| Web FW | Flask + Flask-SQLAlchemy + Flask-Migrate + Flask-WTF |
| Browser | Playwright (async) + Selenium stealth |
| AI/ML | TensorFlow, PyTorch, rembg, ONNX, Google Generative AI, OpenAI |
| Data | pandas, BeautifulSoup4, icrawler |
| Testing | pytest |

---

## 2. 现状分析

### 2.1 変更履歴（Git Status）
```
M app/agents/browser/navigation_driver.py   # ナビゲーションドライバ
M app/agents/browser_use_agent.py           # ブラウザ使用エージェント
M instance/llm_cache/cache.db               # LLMキャッシュ
M tests/test_orchestrator.py                # オーケストレータテスト
M tests/test_rembg.py                       # rembgテスト
```

### 2.2 主要コンポーネント现状

| コンポーネント | 版本 | 状態 | 備考 |
|-------------|------|------|------|
| BrowserUseAgent | v88.6.2J | 開発中 | Moncler PLP対応済み |
| NavigationDriver | 新規 | 開発中 | PLP materialize を移管 |
| MonclerPLPStrategy | v1 | 開発中 | セレクタ・OneTrust対応 |
| AiResearchOrchestrator | v8.0.0J | 安定 | 最高司令部 |
| SupplierScoutAgent | - | 開発中 | 偵察指揮官 |

### 2.3 既知の課題
1. **Moncler PLP が正常に materialise しない場合がある**
   - ロケールトラップ（/en-jp/, /client-service/contact/）
   - OneTrust GDPR バナー対応
   - タイルセレクタの不一致

2. **テストの安定性**
   - `test_orchestrator.py` がヘッドフルモードで実行される
   - 外部API依存がある

---

## 3. 開発方針

### 3.1 基本原則
1. **小さなdiff**: 1回の操作で触るファイル数を最小限に
2. **拡張・オプション追加优先**: 破壊的変更よりも拡張を优先
3. **テスト重視**: テストを追加・修正してから本番コードを变更
4. **日本語コミュニケーション**: すべての応答・文档は日本語

### 3.2 優先度高いタスク

#### P0 - 今すぐ対応
| タスク | ファイル | 概要 |
|--------|---------|------|
| テスト安定化 | `tests/test_orchestrator.py` | ヘッドフル→ヘッドレス切换 |
| テスト安定化 | `tests/test_rembg.py` | rembg 功能测试 |
| navigation_driver 改良 | `app/agents/browser/navigation_driver.py` | PLP materialize 强化 |

#### P1 - 近期対応
| タスク | ファイル | 概要 |
|--------|---------|------|
| Moncler PLP 安定化 | `app/agents/plugins/moncler_plp_v1.py` | セレクタ最適化 |
| FKB (Failure Knowledge Base) | `app/agents/fkb_local.json` | 失敗パターンの蓄積 |
| Auto-Heal 强化 | `app/agents/self_healing_agent.py` | 自動修復机能强化 |

#### P2 - 中期対応
| タスク | ファイル | 概要 |
|--------|---------|------|
| 他のサイト対応 | `app/agents/plugins/` | SSENSE、Farfetchなど |
| ダッシュボード改善 | `app/web/dashboard.py` | UI/UX改善 |
| 収益性分析强化 | `app/agents/profitability_agent.py` | 利益計算精度向上 |

### 3.3 開発プロセス

```
1. タスク選択
   ↓
2. テスト追加・修正（tests/ のみ）
   ↓
3. 本番コード変更（必要に応じて最小限）
   ↓
4. pytest 実行
   ↓
5. 結果レポート作成
   ↓
6. 完了
```

### 3.4 禁止事項
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
1. テストを実行して现状確認
2. 失敗したテストの分析及と修正方針の決定

### 5.2  результат レポート
テスト実行後、以下の內容を含むレポートを作成：
- 合計テスト数
- 成功/失敗/スキップ数
- 失敗したテストの一覧
- エラーの要約
- 修正の方向性

---

*この仕様書はプロジェクトの進行に応じて更新されます。*
