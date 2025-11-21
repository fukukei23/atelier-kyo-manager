# SUMMARY_PACK — atelier-kyo-manager
- **Profile**: GPT-5 Plain Folder (No-ZIP)
- **Export Mode**: Plain Folder (No-ZIP)

## 1) まずココから（読み順）
- `SUMMARY_PACK.md`（このファイル）
- `PROJECT_CHRONICLE.md`（最近1年の進化の物語）
- `PROJECT_BLUEPRINT.md`（依存の地図 / Mermaid）
- `PROJECT_INFO.md`（統計とディレクトリ構造）
- `source_code/`（元構造のソース一式 / plain-folder or full_context の場合）

## 2) 重要ファイル Top20（推定）
- `atelier-kyo-manager/.venv\Lib\site-packages\gradio\cli\commands\components\files\README.md`  (0 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5FullContextPack_20251026_071005\source_code\atelier-kyo-manager\.venv\Lib\site-packages\gradio\cli\commands\components\files\README.md`  (0 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5PlainFolder_No-ZIP_20251026_082722\source_code\atelier-kyo-manager\.venv\Lib\site-packages\gradio\cli\commands\components\files\README.md`  (0 KB)
- `atelier-kyo-manager/.venv\Lib\site-packages\gradio\_frontend_code\client\README.md`  (13 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5FullContextPack_20251026_071005\source_code\atelier-kyo-manager\.venv\Lib\site-packages\gradio\_frontend_code\client\README.md`  (13 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5PlainFolder_No-ZIP_20251026_082722\source_code\atelier-kyo-manager\.venv\Lib\site-packages\gradio\_frontend_code\client\README.md`  (13 KB)
- `atelier-kyo-manager/.venv\Lib\site-packages\wtforms\locale\README.md`  (1 KB)
- `atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\locale\README.md`  (1 KB)
- `atelier-kyo-manager/export\atelier-kyo-manager_export_20250811_005523\source_code\atelier-kyo-manager\.venv_backup\Lib\site-packages\wtforms\locale\README.md`  (1 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5FullContextPack_20251026_071005\source_code\atelier-kyo-manager\.venv\Lib\site-packages\wtforms\locale\README.md`  (1 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5FullContextPack_20251026_071005\source_code\atelier-kyo-manager\.venv_backup\Lib\site-packages\wtforms\locale\README.md`  (1 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5PlainFolder_No-ZIP_20251026_082722\source_code\atelier-kyo-manager\.venv\Lib\site-packages\wtforms\locale\README.md`  (1 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5PlainFolder_No-ZIP_20251026_082722\source_code\atelier-kyo-manager\.venv_backup\Lib\site-packages\wtforms\locale\README.md`  (1 KB)
- `atelier-kyo-manager/README.md`  (1 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5FullContextPack_20251026_071005\source_code\atelier-kyo-manager\README.md`  (1 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5PlainFolder_No-ZIP_20251026_082722\source_code\atelier-kyo-manager\README.md`  (1 KB)
- `atelier-kyo-manager/.venv\Lib\site-packages\pandas\pyproject.toml`  (24 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5FullContextPack_20251026_071005\source_code\atelier-kyo-manager\.venv\Lib\site-packages\pandas\pyproject.toml`  (24 KB)
- `atelier-kyo-manager/exports_atelier\atelier-kyo-manager_GPT-5PlainFolder_No-ZIP_20251026_082722\source_code\atelier-kyo-manager\.venv\Lib\site-packages\pandas\pyproject.toml`  (24 KB)
- `atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\pyproject.toml`  (24 KB)

## 3) プロジェクト年代記（ダイジェスト）
# 📖 プロジェクト年代記 (AI-Generated)
このドキュメントはGitのコミット履歴を基に、プロジェクトの進化の物語を自動生成したものです。
AIが分析を始める前に、まずこの年代記を読むことで、開発の文脈や意図を深く理解できます。
---

### EPOCH: 2025年09月22日 の週
**テーマ: Fixes & Maintenance**
- chore: stop tracking browser profile large files
- build: .gitignoreを更新し、追跡済みの生成ファイルを除外
- chore: apply pre-commit fixes

### EPOCH: 2025年08月25日 の週
**テーマ: Fixes & Maintenance**
- chore: ignore generated research logs
- chore(pre-commit): NBSP/TAB自動修正＋EOF整形＋vendor除外を導入

### EPOCH: 2025年08月18日 の週
**テーマ: General Updates**
- Initial commit

## 4) 設計図（ダイジェスト）
# 🗺️ プロジェクト設計図 (AI-Generated)
このドキュメントはPythonモジュール間の依存関係を可視化したものです。
AIは、この設計図からプロジェクトの全体構造を把握し、より高レベルな分析を行うことができます。
---
```mermaid
graph TD;
    "buyma_research_tool.py" --> "pricing_calculator.py";
    "export/atelier-kyo-manager_export_20250811_005523/combined_98.py" --> "pricing_calculator.py";
    "export/atelier-kyo-manager_export_20250811_005523/source_code/atelier-kyo-manager/buyma_research_tool.py" --> "pricing_calculator.py";
```

## 使い方
この要約パックは、まず全体像を掴むための**最短ルート**です。
詳細やフルソースは同ディレクトリの `PROJECT_*.md` や `source_code/` を参照してください。