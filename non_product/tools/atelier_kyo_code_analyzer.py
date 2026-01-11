# ======================================================================
# ファイル名: atelier_kyo_code_analyzer.py
# 場所: C:\Users\USER\tools\atelier-kyo-manager\tools\
# 役割: 「atelier-kyo-manager」プロジェクト専用の適応型コード分析・エクスポートツール
# 更新日: 2025-08-25 (v4)
#
# v4変更点:
# - 【新機能】AGENT_ARCHITECTURE.md (エージェント設計図) の自動生成機能を追加。
#   AI Analysis Pack選択時、専門家エージェントの役割と連携関係を分析・可視化します。
# - 【改良】COMBINED_CODE.py を構造化。各コードブロックの前にエージェントの役割要約を挿入。
# - これにより、AIは「物語(年代記)」「設計図(アーキテクチャ)」「実装(コード)」の
#   3つの階層でプロジェクトを深く、かつ体系的に理解できるようになります。
#
# 概要:
# UIから選択した「プロファイル」に応じて、出力形式と内容が動的に変化する。
# - [Developer Quick-Share]: 開発者間の情報共有に最適化
# - [AI Analysis Pack (for Gemini)]: 【物語+設計図付き】Geminiでの分析に最適化
# - [Production Build Audit]: 本番ビルドの監査に必要なファイルを網羅
# - [Security & Compliance Review]: セキュリティ関連ファイルを中心に抽出
#
# 使用方法:
# 1. (venv)環境で `python atelier_kyo_code_analyzer.py` を実行
# 2. ブラウザで http://127.0.0.1:7862 にアクセス
# 3. AI Analysis Packを選択すると、Git履歴とコード構造が解析され、
#    年代記と設計図が同梱された、よりリッチな分析パッケージが生成されます。
# ======================================================================

from __future__ import annotations

import ast
import datetime
import fnmatch
import os
import re
import time
import traceback
import zipfile
import subprocess
from collections import defaultdict
from math import log2
from pathlib import Path
from threading import Event
from typing import List, Tuple, Dict, Set, Optional, Any

import gradio as gr

try:
    import networkx as nx
except ImportError:
    nx = None

# --- グローバル設定 ---
DEFAULT_PROJECT_PATH = r"C:\Users\USER\tools\atelier-kyo-manager"

# --- プロファイル定義 ---
PROFILES = {
    "Developer Quick-Share": {
        "description": "開発者間のコード共有に最適化。主要なソースコードと設定ファイルを含みます。",
        "keep": {
            "requirements.txt", "pyproject.toml", ".env.template", "README.md",
            "app.py", "main.py", "vite.config.js", "package.json",
        },
        "include_dirs": set(),
        "export_mode": "standard",
    },
    "AI Analysis Pack (for Gemini)": {
        "description": "【物語+設計図付き】Geminiでの分析に最適化。Git履歴からプロジェクトの進化の物語を、コード構造からエージェントの設計図を自動生成し、ソースコードと結合します。",
        "keep": {
            "requirements.txt", "pyproject.toml", ".env.template", "README.md",
            "ARCHITECTURE.md", "openapi.yaml",
        },
        "include_dirs": set(),
        "export_mode": "gemini",
    },
    "Production Build Audit": {
        "description": "本番環境へのデプロイ構成を監査。DockerfileやCI/CD関連ファイルを網羅します。",
        "keep": {
            "requirements.txt", "pyproject.toml", ".env.template", "README.md",
            "poetry.lock", "uv.lock", "Dockerfile", "docker-compose.yml",
            "Makefile", ".gitlab-ci.yml",
        },
        "include_dirs": {"migrations", "infra", "deploy", "k8s", ".github/workflows"},
        "export_mode": "standard",
    },
    "Security & Compliance Review": {
        "description": "セキュリティとコンプライアンスのレビュー用。認証、設定、依存関係のファイルに焦点を当てます。",
        "keep": {
            "requirements.txt", "pyproject.toml", ".env.template", "poetry.lock",
            "uv.lock", "package-lock.json", "yarn.lock", "alembic.ini",
            "THIRD_PARTY_NOTICES.md", "LICENSE",
        },
        "include_dirs": {"infra", "deploy", ".github"},
        "export_mode": "standard",
    },
}
DEFAULT_PROFILE = "AI Analysis Pack (for Gemini)"

# --- 除外ルールと重み付け ---
BASE_IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", "htmlcov", ".gradio", "exports",
}
BASE_IGNORED_FILES = {
    ".coverage", ".env", ".gitattributes", ".gitignore",
    "*.log", "*.swp", "*.swo",
}
FILE_TYPES: Tuple[str, ...] = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".ipynb", ".md", ".txt", "package.json",
    ".json", ".yml", ".yaml", ".toml", ".css", ".html", ".sql",
)
MAX_TOTAL_SIZE_MB = 100
MAX_GEMINI_CODE_SIZE_MB = 10
WIN_PATH_CAP = 250
EXPORT_ROOT = Path("./exports_atelier").resolve()
EXPORT_ROOT.mkdir(exist_ok=True)
BASE_IGNORED_DIRS.add(EXPORT_ROOT.name)

EXT_WEIGHT = {".py": 3, ".js": 2, ".ts": 2, ".tsx": 2, ".sql": 2, ".md": 1}
PATH_WEIGHT = {"src": 2, "app": 2, "components": 2, "lib": 1, "tests": -1, "docs": -1}

stop_event = Event()


# ======================================================================
# 【v3】プロジェクト年代記ジェネレータ
# ======================================================================
class ChronicleGenerator:
    """Gitのコミット履歴を解析し、プロジェクトの年代記を生成するクラス"""
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.theme_keywords = {
            "Architecture & Refactoring": ["refactor", "archi", "structure", "reorganize", "cleanup"],
            "AI & Agents": ["ai", "agent", "prompt", "model", "llm", "gemini"],
            "Features & UI": ["feat", "ui", "gui", "add", "implement", "ux", "component"],
            "Fixes & Maintenance": ["fix", "bug", "hotfix", "ci", "test", "deps", "typo", "chore"],
            "Documentation": ["docs", "readme", "comment", "document"],
        }

    def _run_git_log(self) -> List[Tuple[datetime.datetime, str]]:
        """git logコマンドを実行し、コミット履歴を取得する"""
        if not (self.repo_path / ".git").is_dir():
            return []
        try:
            one_year_ago = (datetime.datetime.now() - datetime.timedelta(days=365)).isoformat()
            cmd = [
                "git", "log", f"--since={one_year_ago}",
                '--pretty=format:%cI|%s'
            ]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True, encoding='utf-8')

            commits = []
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    date_str, message = line.split('|', 1)
                    try:
                        commits.append((datetime.datetime.fromisoformat(date_str), message))
                    except ValueError:
                        continue
            return commits
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    def _group_by_week(self, commits: List[Tuple[datetime.datetime, str]]) -> Dict[datetime.date, List[str]]:
        """コミットを週ごとにグループ化する"""
        weekly_commits = defaultdict(list)
        for date, message in commits:
            start_of_week = date.date() - datetime.timedelta(days=date.weekday())
            weekly_commits[start_of_week].append(message)
        return weekly_commits

    def _determine_theme(self, messages: List[str]) -> str:
        """コミットメッセージから週のテーマを判定する"""
        scores = defaultdict(int)
        for msg in messages:
            for theme, keywords in self.theme_keywords.items():
                if any(kw in msg.lower() for kw in keywords):
                    scores[theme] += 1
        return max(scores, key=scores.get) if scores else "General Updates"

    def generate(self) -> Optional[str]:
        """プロジェクト年代記のMarkdownコンテンツを生成する"""
        commits = self._run_git_log()
        if not commits:
            return None

        weekly_commits = self._group_by_week(commits)

        md_lines = ["# 📖 プロジェクト年代記 (AI-Generated)",
                    "このドキュメントはGitのコミット履歴を基に、プロジェクトの進化の物語を自動生成したものです。",
                    "AIが分析を始める前に、まずこの年代記を読むことで、開発の文脈や意図を深く理解できます。", "---"]

        sorted_weeks = sorted(weekly_commits.keys(), reverse=True)[:12] # 直近12週分

        for week_start in sorted_weeks:
            messages = weekly_commits[week_start]
            theme = self._determine_theme(messages)

            md_lines.append(f"\n### EPOCH: {week_start.strftime('%Y年%m月%d日')} の週")
            md_lines.append(f"**テーマ: {theme}**")

            for msg in messages[:3]: # 代表的なコミットを3つ表示
                md_lines.append(f"- {msg}")
            if len(messages) > 3:
                md_lines.append(f"- ...他 {len(messages) - 3} 件の改善")

        return "\n".join(md_lines)


# ======================================================================
# 【v4】エージェント設計図ジェネレータ
# ======================================================================
class ArchitectureAnalyzer:
    """コードベースを静的解析し、エージェントの役割と連携関係を記述した設計図を生成する"""
    def __init__(self, root: Path, all_py_files: List[Path]):
        self.root = root
        self.all_py_files = all_py_files
        self.agent_patterns = [
            "_orchestrator.py", "_agent.py", "_manager.py", "_controller.py",
            "_calculator.py", "_crawler.py", "_generator.py", "_collector.py"
        ]
        self.role_map: Dict[Path, str] = {}
        self.architecture: Dict[str, Any] = {"orchestrators": [], "specialists": []}

    def _is_agent(self, file_path: Path) -> Optional[str]:
        """ファイルが命名規則に基づきエージェントかどうかを判定"""
        for pattern in self.agent_patterns:
            if file_path.name.endswith(pattern):
                return pattern.replace(".py", "").replace("_", " ").title()
        return None

    def _summarize_role(self, file_path: Path) -> str:
        """ファイルのdocstringやコメントから役割を要約"""
        try:
            content = file_path.read_text("utf-8", errors="ignore")
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            if docstring:
                return docstring.strip().split('\n')[0]

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    if class_doc:
                        return f"クラス '{node.name}': {class_doc.strip().splitlines()[0]}"

            # ファイル先頭のコメントブロックを探す
            match = re.search(r"^[#\s-]*役割\s*[:：(](.*?)$", content, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        except Exception:
            pass

        # フォールバック
        return f"{file_path.stem.replace('_', ' ').title()} の機能を提供"

    def _map_interactions(self, orchestrator_path: Path, agent_map: Dict[str, Path]) -> List[str]:
        """司令塔から他のエージェントへの呼び出し関係を簡易的に解析"""
        interactions = []
        try:
            content = orchestrator_path.read_text("utf-8", errors="ignore")
            tree = ast.parse(content)

            imported_agents = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for name in node.names:
                        if node.module.endswith(name.name.lower()): # 簡易的な自己参照回避
                            continue
                        for agent_name, agent_path in agent_map.items():
                            if agent_path.stem == node.module.split('.')[-1]:
                                imported_agents[name.name] = agent_name

            # 簡単なテキスト検索で代用
            for agent_name in imported_agents.values():
                 if agent_name in content:
                     interactions.append(agent_name)

        except Exception:
            pass
        return sorted(list(set(interactions)))


    def analyze(self) -> "ArchitectureAnalyzer":
        """プロジェクト全体のアーキテクチャを解析"""
        agents = {}
        specialists_by_path = {}
        orchestrators_by_path = {}

        # 1. エージェントを識別し、役割を要約
        for file_path in self.all_py_files:
            agent_type = self._is_agent(file_path)
            if agent_type:
                role = self._summarize_role(file_path)
                self.role_map[file_path] = role
                agent_info = {"name": file_path.name, "role": role, "path": file_path}

                if "Orchestrator" in agent_type:
                    self.architecture["orchestrators"].append(agent_info)
                    orchestrators_by_path[file_path.name] = file_path
                else:
                    self.architecture["specialists"].append(agent_info)
                    specialists_by_path[file_path.name] = file_path

        # 2. 司令塔と専門家の連携関係をマッピング
        for orch in self.architecture["orchestrators"]:
            orch["interactions"] = self._map_interactions(orch["path"], specialists_by_path)

        return self

    def generate_markdown(self) -> str:
        """解析結果をMarkdown形式の設計図として出力"""
        if not self.architecture["orchestrators"] and not self.architecture["specialists"]:
            return ""

        lines = [
            "# 🏛️ エージェント設計図 (AI-Generated)",
            "このドキュメントはプロジェクトのソースコードを静的解析し、自律的に動作する「専門家エージェント」の役割と連携関係を可視化したものです。",
            "AIがコードの詳細を分析する前にこの設計図を読むことで、システム全体のアーキテクチャと目的を効率的に理解できます。",
            "---"
        ]

        if self.architecture["orchestrators"]:
            lines.append("\n## 🧠 司令塔 (Orchestrators)")
            lines.append("複数の専門家エージェントを連携させ、複雑なワークフローを実行する中心的役割を担います。")
            for orch in self.architecture["orchestrators"]:
                lines.append(f"\n###  orchestrator: `{orch['name']}`")
                lines.append(f"- **役割:** {orch['role']}")
                if orch.get("interactions"):
                    lines.append("- **連携する専門家:**")
                    for inter in orch["interactions"]:
                        lines.append(f"  - `{inter}`")

        if self.architecture["specialists"]:
            lines.append("\n## 🛠️ 専門家 (Specialists)")
            lines.append("特定のタスク（例：価格計算、画像収集）に特化したエージェントです。司令塔から呼び出されて機能します。")
            for spec in self.architecture["specialists"]:
                lines.append(f"\n- **`{spec['name']}`**")
                lines.append(f"  - **専門分野:** {spec['role']}")

        return "\n".join(lines)


# --- 既存ヘルパー関数群 (変更なし) ---
def sanitize_name(name: str) -> str:
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    s = re.sub(invalid_chars, '_', name)
    s = re.sub(r'_+', '_', s).strip('_') or "Unknown"
    return s[:50]

def generate_project_prefix(roots: List[Path]) -> str:
    if not roots: return "Export"
    names = [sanitize_name(r.name) for r in roots]
    if len(names) == 1: return names[0]
    return "-".join(names) if len(names) <= 3 else f"{'-'.join(names[:2])}-etc{len(names)-2}"

def load_gitignore(root: Path) -> Set[str]:
    gi = root / ".gitignore"
    if not gi.exists(): return set()
    return {
        line.strip()
        for line in gi.read_text("utf-8", errors="ignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

def path_is_ignored(path: Path, root: Path, patterns: Set[str], include_dirs: Set[str]) -> bool:
    for inc_dir in include_dirs:
        if (root / inc_dir) in path.parents:
            return False
    if any((root / d) in path.parents for d in BASE_IGNORED_DIRS):
        return True
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        return True
    for pat in patterns:
        if fnmatch.fnmatch(str(rel_path), pat) or fnmatch.fnmatch(path.name, pat):
            return True
    if any(fnmatch.fnmatch(path.name, pat) for pat in BASE_IGNORED_FILES):
        return True
    return False

def build_import_map(root: Path, py_files: List[Path], progress: gr.Progress) -> Dict[Path, Set[Path]]:
    progress(0.35, desc="import依存関係を解析中...")
    module_of = {".".join(p.relative_to(root).with_suffix("").parts): p for p in py_files}
    mapping: Dict[Path, Set[Path]] = defaultdict(set)
    for i, pf in enumerate(py_files):
        if stop_event.is_set(): break
        try:
            tree = ast.parse(pf.read_text("utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        tgt = module_of.get(alias.name.split(".")[0])
                        if tgt: mapping[pf].add(tgt)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    tgt = module_of.get(node.module.split(".")[0])
                    if tgt: mapping[pf].add(tgt)
        except Exception:
            continue
        if i % 10 == 0 and py_files:
            progress(0.35 + (0.1 * i / len(py_files)), desc=f"import解析中... ({i+1}/{len(py_files)})")
    return mapping

def degree_centrality(mapping: Dict[Path, Set[Path]]) -> Dict[Path, float]:
    if not nx: return {k: float(len(v)) for k, v in mapping.items()}
    g = nx.DiGraph(mapping)
    return nx.degree_centrality(g)

def loc_count(path: Path) -> int:
    try: return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
    except Exception: return 0

def file_score(path: Path, root: Path, cent: Dict[Path, float]) -> float:
    score = EXT_WEIGHT.get(path.suffix.lower(), 0)
    score += next((PATH_WEIGHT[p] for p in path.parts if p in PATH_WEIGHT), 0)
    score += cent.get(path, 0) * 5
    lines = loc_count(path)
    if lines: score += min(log2(lines + 1) / 5, 3)
    return score

def write_redacted_env(src_env: Path, out_dir: Path):
    try:
        lines = src_env.read_text("utf-8", errors="ignore").splitlines()
        out = ["# This file is redacted for security reasons. Original values are replaced with '****'."]
        for ln in lines:
            if not ln.strip() or ln.strip().startswith("#"):
                out.append(ln)
                continue
            key, _, _ = ln.partition("=")
            out.append(f"{key}=****")
        (out_dir / ".env.redacted").write_text("\n".join(out), "utf-8")
    except Exception as e:
        (out_dir / ".env.redacted.error").write_text(f"Failed to redact .env: {e}")

def collect_and_score_files(root: Path, progress: gr.Progress, keep_set: Set[str], include_dirs: Set[str]) -> List[Tuple[float, Path]]:
    progress(0.05, desc=f"スキャン中: {root.name}")
    patterns = load_gitignore(root)
    candidates: List[Path] = []
    try:
        all_entries = list(root.rglob("*"))
    except Exception:
        return []

    for i, p in enumerate(all_entries):
        if stop_event.is_set(): break
        is_kept = p.name in keep_set
        if is_kept or (p.is_file() and p.suffix.lower() in FILE_TYPES and not path_is_ignored(p, root, patterns, include_dirs)):
            candidates.append(p)
        if i % 100 == 0 and all_entries:
            progress(0.1 + (0.2 * i / len(all_entries)), desc=f"ファイル収集中... ({len(candidates)}件)")

    if not candidates: return []

    py_files = [p for p in candidates if p.suffix == ".py"]
    centrality = degree_centrality(build_import_map(root, py_files, progress)) if py_files and nx else {}

    scored_files = []
    for p in candidates:
        score = file_score(p, root, centrality)
        if p.name in keep_set:
            score += 100
        scored_files.append((score, p))

    scored_files.sort(key=lambda x: x[0], reverse=True)
    return scored_files


# --- Export Logics (v4で改良) ---
def export_standard(session_dir: Path, all_files: List[Tuple[Path, Path]], roots: List[Path], profile: str, progress: gr.Progress):
    source_code_dir = session_dir / "source_code"
    source_code_dir.mkdir()

    progress(0.8, desc="ファイルをコピー中...")
    for i, (root, file_path) in enumerate(all_files):
        if stop_event.is_set(): break
        try:
            rel_path = file_path.relative_to(root)
            dest_path = source_code_dir / root.name / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(file_path.read_bytes())
        except Exception:
            continue
        if i % 10 == 0 and all_files:
            progress(0.8 + (0.1 * i / len(all_files)), desc=f"コピー中... ({i+1}/{len(all_files)})")

    info_content = create_project_info_content(
        len(all_files), all_files, roots, profile, "Standard ZIP"
    )
    (session_dir / "project_info.md").write_text(info_content, "utf-8")

def export_gemini(session_dir: Path, all_files: List[Tuple[Path, Path]], roots: List[Path], profile: str, progress: gr.Progress) -> List[str]:
    """Gemini用のエクスポートを行い、アーカイブ対象のファイルリストを返す"""
    progress(0.75, desc="Gemini用ファイル生成中...")

    # 1. 年代記 (Chronicle)
    chronicle_content = ChronicleGenerator(roots[0]).generate()
    if chronicle_content:
        (session_dir / "PROJECT_CHRONICLE.md").write_text(chronicle_content, "utf-8")

    # 2. 【NEW】設計図 (Architecture)
    all_py = [p for _, p in all_files if p.suffix == ".py"]
    arch_analyzer = ArchitectureAnalyzer(roots[0], all_py).analyze()
    architecture_content = arch_analyzer.generate_markdown()
    if architecture_content:
        (session_dir / "AGENT_ARCHITECTURE.md").write_text(architecture_content, "utf-8")

    # 3. プロジェクト情報 (Info)
    info_content = create_project_info_content(
        len(all_files), all_files, roots, profile, "Gemini Optimized"
    )
    (session_dir / "PROJECT_INFO.md").write_text(info_content, "utf-8")

    # 4. 【IMPROVED】構造化・結合コード (Combined Code)
    header = f"""
# ======================================================================
# !! 警告: これはAI分析用の合成ファイルです !!
# WARNING: THIS IS A SYNTHETIC FILE FOR AI ANALYSIS
# ======================================================================
#
# このファイル (`COMBINED_CODE.py`) は、Atelier-Kyo Code Analyzerによって
# 自動生成されました。元のプロジェクトの複数のソースファイルを、分析しやすい
# ように1つのファイルに結合したものです。
#
# !! このファイルを直接編集したり、プロジェクトの実際の構造だと解釈しないでください !!
#
# 本来のプロジェクトは、多数のファイルから成るモジュール化された構造です。
# 正確なディレクトリ構造やファイル一覧については、このZIPアーカイブに
# 同梱されている `PROJECT_INFO.md` を、エージェントの役割と連携については
# `AGENT_ARCHITECTURE.md` を参照してください。
#
# 目的:
#   大規模言語モデル(LLM)がプロジェクトの全体像を一度に把握し、
#   より精度の高い分析や提案を行えるようにするため。
#
# 生成日時: {datetime.datetime.now().isoformat()}
# 結合ファイル数: {len(all_files)}
#
# --- 結合されたファイル一覧 ---
"""
    combined_lines = [header.strip()]
    for i, (root, file_path) in enumerate(all_files):
        rel_path_str = file_path.name
        try: rel_path_str = f"{root.name}/{file_path.relative_to(root)}"
        except ValueError: pass
        combined_lines.append(f"# {i+1:03d}: {rel_path_str}")
    combined_lines.append("# ======================================================================\n")

    for i, (root, file_path) in enumerate(all_files):
        if stop_event.is_set(): break
        rel_path_str = file_path.name
        try: rel_path_str = f"{root.name}/{file_path.relative_to(root)}"
        except ValueError: pass

        # --- v4 改良点: 役割ヘッダーの挿入 ---
        role_header = "#" + "="*70
        role_header += f"\n# --- FILE: {rel_path_str} ---\n"
        role = arch_analyzer.role_map.get(file_path)
        if role:
            role_header += f"# ROLE: {role}\n"
        role_header += "#" + "="*70 + "\n"
        combined_lines.append(f"\n{role_header}")
        # --- ここまで ---

        try:
            combined_lines.append(file_path.read_text("utf-8", "ignore"))
        except Exception as e:
            combined_lines.append(f"# FAILED to read {file_path}: {e}")
        combined_lines.append(f"\n# --- END OF {file_path.name} ---\n")

    (session_dir / "COMBINED_CODE.py").write_text("\n".join(combined_lines), "utf-8")

    # 5. README
    readme = f"""# AI Analysis Pack for {generate_project_prefix(roots)}
This ZIP is optimized for analysis by AI models like Gemini. It contains a multi-layered view of the project:

1.  **`PROJECT_CHRONICLE.md` (The Story - Start here!)**
    * An AI-generated summary of the project's history from Git commits.
    * **Recommendation: Read this first** to understand the project's evolution and development context.

2.  **`AGENT_ARCHITECTURE.md` (The Blueprint - NEW!)**
    * An AI-generated blueprint of the system, identifying key "Agent" components, their roles, and how they interact.
    * **Recommendation: Read this second** to grasp the overall system design.

3.  **`PROJECT_INFO.md` (The Stats)**
    * Contains project statistics, file counts by type, and the full directory structure.

4.  **`COMBINED_CODE.py` (The Implementation - IMPROVED!)**
    * A synthetic file where all source code is concatenated. Each file's code is now prefixed with a header summarizing its role, making it easier to navigate.

5.  **`README.md` (This file)**
"""
    (session_dir / "README.md").write_text(readme, "utf-8")

    # ZIPに含めるファイルリストを返す
    archive_files = ["README.md", "PROJECT_INFO.md", "COMBINED_CODE.py"]
    if architecture_content:
        archive_files.insert(0, "AGENT_ARCHITECTURE.md")
    if chronicle_content:
        archive_files.insert(0, "PROJECT_CHRONICLE.md")
    return archive_files


def create_project_info_content(file_count: int, all_files: List[Tuple[Path, Path]], roots: List[Path], profile: str, mode: str) -> str:
    total_lines = 0
    stats = defaultdict(lambda: {"files": 0, "lines": 0})
    for _, p in all_files:
        ext = p.suffix or "other"
        lines = loc_count(p)
        stats[ext]["files"] += 1
        stats[ext]["lines"] += lines
        total_lines += lines

    tree_parts = []
    for root in roots:
        root_files = [p for r, p in all_files if r == root]
        if not root_files: continue

        tree_dict = {}
        for f in root_files:
            node = tree_dict
            try:
                parts = f.relative_to(root).parts
                for part in parts:
                    node = node.setdefault(part, {})
            except ValueError:
                continue

        lines = [f"📁 **{root.name}/**"]
        def build_tree(d, prefix=""):
            items = sorted(d.keys())
            for i, name in enumerate(items):
                connector = "└── " if i == len(items) - 1 else "├── "
                lines.append(f"{prefix}{connector}{name}")
                if d[name]:
                    new_prefix = prefix + ("    " if i == len(items) - 1 else "│   ")
                    build_tree(d[name], new_prefix)
        build_tree(tree_dict)
        tree_parts.append("\n".join(lines))

    return f"""# Project Analysis Report
- **Project**: {generate_project_prefix(roots)}
- **Profile**: {profile}
- **Export Mode**: {mode}
- **Timestamp**: {datetime.datetime.now().isoformat()}

## 📊 Statistics
- **Total Files**: {file_count}
- **Total Lines**: {total_lines:,}

| Extension | Files | Lines of Code |
|---|---|---|
{"".join(f"| `{ext}` | {data['files']:,} | {data['lines']:,} |\n" for ext, data in sorted(stats.items(), key=lambda item: item[1]['files'], reverse=True))}

## 🌳 Directory Structure
```
{chr(10).join(tree_parts)}
```
"""

# --- Main Logic ---
def run_export(roots_str: str, profile_name: str, redact_env: bool, progress=gr.Progress(track_tqdm=True)):
    stop_event.clear()
    roots_list = [p.strip() for p in roots_str.splitlines() if p.strip()]
    if not roots_list: return None, "⚠️ 少なくとも1つのフォルダを選択してください"

    roots = [Path(p).expanduser().resolve() for p in roots_list]
    if not all(r.is_dir() for r in roots): return None, "⚠️ 無効なフォルダパスが含まれています"

    profile = PROFILES[profile_name]
    keep_set, include_dirs = set(profile["keep"]), set(profile["include_dirs"])
    export_mode = profile["export_mode"]
    max_size_bytes = (MAX_GEMINI_CODE_SIZE_MB if export_mode == 'gemini' else MAX_TOTAL_SIZE_MB) * 1024 * 1024

    try:
        all_scored_files = []
        for i, root in enumerate(roots):
            progress(i / len(roots) * 0.1, desc=f"プロジェクト解析中: {root.name}")
            class SubProgress:
                def __call__(self, value, desc=""):
                    base = i / len(roots) * 0.7
                    scale = 0.7 / len(roots)
                    progress(base + value * scale, desc=desc)
            scored = collect_and_score_files(root, SubProgress(), keep_set, include_dirs)
            all_scored_files.extend([(score, root, path) for score, path in scored])

        all_scored_files.sort(key=lambda x: x[0], reverse=True)

        final_files: List[Tuple[Path, Path]] = []
        final_paths: Set[Path] = set()
        current_size = 0
        for _, root, path in all_scored_files:
            if path in final_paths: continue
            try:
                size = path.stat().st_size
                if current_size + size <= max_size_bytes:
                    final_files.append((root, path))
                    final_paths.add(path)
                    current_size += size
            except FileNotFoundError:
                continue

        if not final_files: return None, "⚠️ 対象ファイルが見つかりませんでした"

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = EXPORT_ROOT / f"{generate_project_prefix(roots)}_{profile_name.replace(' ', '')}_{ts}"
        session_dir.mkdir(exist_ok=True)

        if redact_env:
            for root in roots:
                if (root / ".env").exists():
                    write_redacted_env(root / ".env", session_dir)

        files_to_zip = []
        if export_mode == 'gemini':
            files_to_zip = export_gemini(session_dir, final_files, roots, profile_name, progress)
        else:
            export_standard(session_dir, final_files, roots, profile_name, progress)
            for f in session_dir.rglob("*"):
                if f.is_file():
                    files_to_zip.append(f.relative_to(session_dir))

        progress(0.95, desc="ZIPアーカイブを作成中...")
        zip_path = session_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if redact_env and (session_dir / ".env.redacted").exists():
                 zf.write(session_dir / ".env.redacted", arcname=".env.redacted")

            if export_mode == 'gemini':
                 for filename in files_to_zip:
                    zf.write(session_dir / filename, arcname=filename)
            else: # standard
                for f in session_dir.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(session_dir))

        mb = zip_path.stat().st_size / (1024 * 1024)
        return str(zip_path), f"✅ エクスポート完了 ({profile_name})\n- サイズ: {mb:.2f} MB\n- 出力先: {zip_path}"

    except Exception as e:
        return None, f"❌ エラー:\n{e}\n\n詳細:\n{traceback.format_exc(limit=3)}"

def cancel_export():
    stop_event.set()
    return "⏹️ キャンセル信号を送信しました"

# --- Gradio UI ---
def create_interface():
    with gr.Blocks(title="Atelier-Kyo Code Analyzer", theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("# 🎨 Atelier-Kyo Code Analyzer (v4: 物語+設計図 生成機能付き)")
        gr.Markdown("プロジェクトの目的に応じて、最適な形でコードを分析・パッケージ化します。**AI Analysis Pack**ではGit履歴からプロジェクトの「物語」を、コード構造から「設計図」を自動生成します。")

        with gr.Row():
            dirs_tb = gr.Textbox(
                label="📁 プロジェクトフォルダ",
                value=DEFAULT_PROJECT_PATH,
                lines=2,
            )
            profile_dd = gr.Dropdown(
                label="🎯 分析プロファイル",
                choices=list(PROFILES.keys()),
                value=DEFAULT_PROFILE,
            )

        profile_info = gr.Markdown(f"**説明:** {PROFILES[DEFAULT_PROFILE]['description']}")

        def update_profile_info(profile_name):
            return gr.update(value=f"**説明:** {PROFILES[profile_name]['description']}")

        profile_dd.change(update_profile_info, inputs=profile_dd, outputs=profile_info)

        redact_ck = gr.Checkbox(label="🔑 .env ファイルをマスクして同梱 (.env.redacted)", value=True)

        with gr.Row():
            exp_btn = gr.Button("🚀 エクスポート開始", variant="primary", size="lg")
            cancel_btn = gr.Button("⏹️ キャンセル")

        zip_out = gr.File(label="📦 ダウンロード")
        status = gr.Textbox(label="📋 ステータス", lines=5, show_copy_button=True)

        export_event = exp_btn.click(
            fn=run_export,
            inputs=[dirs_tb, profile_dd, redact_ck],
            outputs=[zip_out, status],
            show_progress="full"
        )
        cancel_btn.click(fn=cancel_export, cancels=[export_event], outputs=[status])

        demo.queue(default_concurrency_limit=1)
    return demo

if __name__ == "__main__":
    print("🎨 Atelier-Kyo Code Analyzer (v4) 起動中...")
    print(f"📁 エクスポート先: {EXPORT_ROOT}")

    allowed_paths = [str(EXPORT_ROOT), DEFAULT_PROJECT_PATH]

    app = create_interface()
    app.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7862,
        allowed_paths=allowed_paths,
        share=False
    )
