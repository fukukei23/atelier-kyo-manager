# ======================================================================
# ファイル名: atelier_kyo_code_analyzer.py
# 場所: C:\Users\USER\tools\atelier-kyo-manager\tools\
# 役割: 「atelier-kyo-manager」プロジェクト専用の適応型コード分析・エクスポートツール
# 更新日: 2025-08-24 (v5)
#
# v5変更点:
# - 【哲学の進化】GPT-5向けプロファイルを「結合ファイル」から「構造化ソースコード」に変更。
#   COMBINED_CODE.pyを廃止し、元のディレクトリ構造を維持した`source_code/`を同梱。
# - GPT-5 Full Context Packプロファイルのサイズ上限を50MBに大幅緩和。
# - AIへのガイダンスを強化し、年代記→設計図→ソースコードという最適な分析手順を提示。
#
# 概要:
# - [GPT-5 Full Context Pack]: 【NEW】次世代AI向け。年代記+設計図+構造化ソースコード一式を提供。
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
    import gradio_client.utils as gc_utils
    _orig_json_schema_to_python_type = gc_utils.json_schema_to_python_type
    def _safe_json_schema_to_python_type(schema: Any, defs=None):
        # Gradio 4 / client が bool スキーマを渡すケースへの保険
        if isinstance(schema, bool):
            return "Any"
        try:
            return _orig_json_schema_to_python_type(schema, defs)
        except Exception:
            return "Any"
    gc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type
except Exception:
    pass

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
        "keep": {"requirements.txt", "pyproject.toml", ".env.template", "README.md", "app.py", "main.py", "vite.config.js", "package.json"},
        "include_dirs": set(),
        "export_mode": "standard",
    },
    "AI Analysis Pack (for Gemini)": {
        "description": "【年代記生成機能付き】Geminiでの分析に最適化。Git履歴からプロジェクトの進化の物語を自動生成します。",
        "keep": {"requirements.txt", "pyproject.toml", ".env.template", "README.md", "ARCHITECTURE.md", "openapi.yaml"},
        "include_dirs": set(),
        "export_mode": "gemini",
    },
    "GPT-5 Full Context Pack": {
        "description": "【NEW: 年代記+設計図+構造化ソース】次世代AI向け。結合ファイルではなく、元の構造を維持したソースコード一式を提供します。",
        "keep": {"requirements.txt", "pyproject.toml", ".env.template", "README.md", "ARCHITECTURE.md", "openapi.yaml"},
        "include_dirs": set(),
        "export_mode": "full_context", # 新しいモード
    },
    "Production Build Audit": {
        "description": "本番環境へのデプロイ構成を監査。DockerfileやCI/CD関連ファイルを網羅します。",
        "keep": {"requirements.txt", "pyproject.toml", ".env.template", "README.md", "poetry.lock", "uv.lock", "Dockerfile", "docker-compose.yml", "Makefile", ".gitlab-ci.yml"},
        "include_dirs": {"migrations", "infra", "deploy", "k8s", ".github/workflows"},
        "export_mode": "standard",
    },
    "Security & Compliance Review": {
        "description": "セキュリティとコンプライアンスのレビュー用。認証、設定、依存関係のファイルに焦点を当てます。",
        "keep": {"requirements.txt", "pyproject.toml", ".env.template", "poetry.lock", "uv.lock", "package-lock.json", "yarn.lock", "alembic.ini", "THIRD_PARTY_NOTICES.md", "LICENSE"},
        "include_dirs": {"infra", "deploy", ".github"},
        "export_mode": "standard",
    },
}
DEFAULT_PROFILE = "GPT-5 Full Context Pack"

# --- 除外ルールと重み付け ---
BASE_IGNORED_DIRS = {".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv", ".idea", ".vscode", ".mypy_cache", ".pytest_cache", "htmlcov", ".gradio", "exports"}
BASE_IGNORED_FILES = {".coverage", ".env", ".gitattributes", ".gitignore", "*.log", "*.swp", "*.swo"}
FILE_TYPES: Tuple[str, ...] = (".py", ".js", ".ts", ".tsx", ".jsx", ".ipynb", ".md", ".txt", "package.json", ".json", ".yml", ".yaml", ".toml", ".css", ".html", ".sql")
MAX_TOTAL_SIZE_MB = 100
MAX_AI_CODE_SIZE_MB = 10
MAX_GPT5_CODE_SIZE_MB = 50 # NEW: GPT-5向けサイズ上限
WIN_PATH_CAP = 250
EXPORT_ROOT = Path("./exports_atelier").resolve()
EXPORT_ROOT.mkdir(exist_ok=True)
BASE_IGNORED_DIRS.add(EXPORT_ROOT.name)
EXT_WEIGHT = {".py": 3, ".js": 2, ".ts": 2, ".tsx": 2, ".sql": 2, ".md": 1}
PATH_WEIGHT = {"src": 2, "app": 2, "components": 2, "lib": 1, "tests": -1, "docs": -1}
stop_event = Event()

# (ChronicleGenerator, BlueprintGenerator, and other helpers remain the same)
class ChronicleGenerator:
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
        if not (self.repo_path / ".git").is_dir(): return []
        try:
            one_year_ago = (datetime.datetime.now() - datetime.timedelta(days=365)).isoformat()
            cmd = ["git", "log", f"--since={one_year_ago}", '--pretty=format:%cI|%s']
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True, encoding='utf-8')
            commits = []
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    date_str, message = line.split('|', 1)
                    try: commits.append((datetime.datetime.fromisoformat(date_str), message))
                    except ValueError: continue
            return commits
        except (subprocess.CalledProcessError, FileNotFoundError): return []
    def _group_by_week(self, commits: List[Tuple[datetime.datetime, str]]) -> Dict[datetime.date, List[str]]:
        weekly_commits = defaultdict(list)
        for date, message in commits:
            start_of_week = date.date() - datetime.timedelta(days=date.weekday())
            weekly_commits[start_of_week].append(message)
        return weekly_commits
    def _determine_theme(self, messages: List[str]) -> str:
        scores = defaultdict(int)
        for msg in messages:
            for theme, keywords in self.theme_keywords.items():
                if any(kw in msg.lower() for kw in keywords): scores[theme] += 1
        return max(scores, key=scores.get) if scores else "General Updates"
    def generate(self) -> Optional[str]:
        commits = self._run_git_log()
        if not commits: return None
        weekly_commits = self._group_by_week(commits)
        md_lines = ["# 📖 プロジェクト年代記 (AI-Generated)", "このドキュメントはGitのコミット履歴を基に、プロジェクトの進化の物語を自動生成したものです。", "AIが分析を始める前に、まずこの年代記を読むことで、開発の文脈や意図を深く理解できます。", "---"]
        sorted_weeks = sorted(weekly_commits.keys(), reverse=True)[:12]
        for week_start in sorted_weeks:
            messages = weekly_commits[week_start]
            theme = self._determine_theme(messages)
            md_lines.append(f"\n### EPOCH: {week_start.strftime('%Y年%m月%d日')} の週")
            md_lines.append(f"**テーマ: {theme}**")
            for msg in messages[:3]: md_lines.append(f"- {msg}")
            if len(messages) > 3: md_lines.append(f"- ...他 {len(messages) - 3} 件の改善")
        return "\n".join(md_lines)

class BlueprintGenerator:
    def __init__(self, graph: "nx.DiGraph", root: Path, included_files: Set[Path]):
        self.graph = graph
        self.root = root
        self.included_files = {p.relative_to(root) for p in included_files if p.suffix == '.py'}
    def generate(self) -> Optional[str]:
        if not nx or not self.graph: return None
        md_lines = ["# 🗺️ プロジェクト設計図 (AI-Generated)", "このドキュメントはPythonモジュール間の依存関係を可視化したものです。", "AIは、この設計図からプロジェクトの全体構造を把握し、より高レベルな分析を行うことができます。", "---", "```mermaid", "graph TD;"]
        nodes_in_blueprint = set()
        for u, v in self.graph.edges():
            try:
                u_rel, v_rel = u.relative_to(self.root), v.relative_to(self.root)
                if u_rel in self.included_files and v_rel in self.included_files:
                    u_node, v_node = str(u_rel).replace("\\", "/"), str(v_rel).replace("\\", "/")
                    md_lines.append(f'    "{u_node}" --> "{v_node}";')
                    nodes_in_blueprint.add(u_node); nodes_in_blueprint.add(v_node)
            except (ValueError, AttributeError): continue
        for node in nodes_in_blueprint:
            if 'main' in node or 'app' in node: md_lines.append(f'    style "{node}" fill:#d4edda,stroke:#155724,stroke-width:2px;')
            elif 'agent' in node or 'ai' in node: md_lines.append(f'    style "{node}" fill:#cce5ff,stroke:#004085,stroke-width:2px;')
            elif 'util' in node or 'lib' in node: md_lines.append(f'    style "{node}" fill:#f8d7da,stroke:#721c24,stroke-width:2px;')
        md_lines.append("```")
        return "\n".join(md_lines) if len(md_lines) > 4 else None

def sanitize_name(name: str) -> str:
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'; s = re.sub(invalid_chars, '_', name)
    return re.sub(r'_+', '_', s).strip('_') or "Unknown"
def generate_project_prefix(roots: List[Path]) -> str:
    if not roots: return "Export"
    names = [sanitize_name(r.name) for r in roots]
    if len(names) == 1: return names[0]
    return "-".join(names) if len(names) <= 3 else f"{'-'.join(names[:2])}-etc{len(names)-2}"
def load_gitignore(root: Path) -> Set[str]:
    gi = root / ".gitignore";
    if not gi.exists(): return set()
    return { line.strip() for line in gi.read_text("utf-8", errors="ignore").splitlines() if line.strip() and not line.lstrip().startswith("#")}
def path_is_ignored(path: Path, root: Path, patterns: Set[str], include_dirs: Set[str]) -> bool:
    for inc_dir in include_dirs:
        if (root / inc_dir) in path.parents: return False
    if any((root / d) in path.parents for d in BASE_IGNORED_DIRS): return True
    try: rel_path = path.relative_to(root)
    except ValueError: return True
    for pat in patterns:
        if fnmatch.fnmatch(str(rel_path), pat) or fnmatch.fnmatch(path.name, pat): return True
    if any(fnmatch.fnmatch(path.name, pat) for pat in BASE_IGNORED_FILES): return True
    return False
def build_import_map(root: Path, py_files: List[Path]) -> Dict[Path, Set[Path]]:
    module_of = {".".join(p.relative_to(root).with_suffix("").parts): p for p in py_files}
    mapping: Dict[Path, Set[Path]] = defaultdict(set)
    for pf in py_files:
        if stop_event.is_set(): break
        try:
            tree = ast.parse(pf.read_text("utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        tgt = module_of.get(alias.name.split(".")[0]);
                        if tgt: mapping[pf].add(tgt)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    tgt = module_of.get(node.module.split(".")[0]);
                    if tgt: mapping[pf].add(tgt)
        except Exception: continue
    return mapping
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
            if not ln.strip() or ln.strip().startswith("#"): out.append(ln); continue
            key, _, _ = ln.partition("="); out.append(f"{key}=****")
        (out_dir / ".env.redacted").write_text("\n".join(out), "utf-8")
    except Exception as e:
        (out_dir / ".env.redacted.error").write_text(f"Failed to redact .env: {e}")
def collect_and_score_files(root: Path, progress: gr.Progress, keep_set: Set[str], include_dirs: Set[str]) -> Tuple[List[Tuple[float, Path]], "nx.DiGraph"]:
    progress(0.05, desc=f"スキャン中: {root.name}")
    patterns = load_gitignore(root)
    candidates: List[Path] = []
    try: all_entries = list(root.rglob("*"))
    except Exception: return [], None
    for i, p in enumerate(all_entries):
        if stop_event.is_set(): break
        is_kept = p.name in keep_set
        if is_kept or (p.is_file() and p.suffix.lower() in FILE_TYPES and not path_is_ignored(p, root, patterns, include_dirs)): candidates.append(p)
        if i % 100 == 0 and all_entries: progress(0.1 + (0.2 * i / len(all_entries)), desc=f"ファイル収集中... ({len(candidates)}件)")
    if not candidates: return [], None
    py_files = [p for p in candidates if p.suffix == ".py"]
    graph, centrality = None, {}
    if nx and py_files:
        progress(0.35, desc="import依存関係を解析中...")
        import_map = build_import_map(root, py_files)
        graph = nx.DiGraph()
        for u, vs in import_map.items():
            for v in vs: graph.add_edge(u, v)
        centrality = nx.degree_centrality(graph)
    scored_files = []
    for p in candidates:
        score = file_score(p, root, centrality)
        if p.name in keep_set: score += 100
        scored_files.append((score, p))
    scored_files.sort(key=lambda x: x[0], reverse=True)
    return scored_files, graph

# --- Export Logics ---
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
        except Exception: continue
        if i % 10 == 0 and all_files: progress(0.8 + (0.1 * i / len(all_files)), desc=f"コピー中... ({i+1}/{len(all_files)})")
    info_content = create_project_info_content(len(all_files), all_files, roots, profile, "Standard ZIP")
    (session_dir / "project_info.md").write_text(info_content, "utf-8")

def export_ai_pack(session_dir: Path, all_files: List[Tuple[Path, Path]], roots: List[Path], profile: str, progress: gr.Progress, dependency_graph: "nx.DiGraph", export_mode: str):
    progress(0.75, desc="AI分析パッケージを生成中...")
    files_to_archive = []
    # 1. 年代記
    chronicle_content = ChronicleGenerator(roots[0]).generate()
    if chronicle_content:
        (session_dir / "PROJECT_CHRONICLE.md").write_text(chronicle_content, "utf-8")
        files_to_archive.append("PROJECT_CHRONICLE.md")
    # 2. 設計図
    if export_mode == "full_context":
        blueprint_content = BlueprintGenerator(dependency_graph, roots[0], {p for _, p in all_files}).generate()
        if blueprint_content:
            (session_dir / "PROJECT_BLUEPRINT.md").write_text(blueprint_content, "utf-8")
            files_to_archive.append("PROJECT_BLUEPRINT.md")
    # 3. プロジェクト情報
    info_content = create_project_info_content(len(all_files), all_files, roots, profile, "AI Optimized")
    (session_dir / "PROJECT_INFO.md").write_text(info_content, "utf-8")
    files_to_archive.append("PROJECT_INFO.md")
    # 4. ソースコード (モードによって形式が変わる)
    if export_mode == "full_context":
        source_code_dir = session_dir / "source_code"
        for root, file_path in all_files:
            dest_path = source_code_dir / root.name / file_path.relative_to(root)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(file_path.read_bytes())
        # source_codeディレクトリ全体をアーカイブ対象にする必要はない (zip処理で対応)
    else: # geminiモード
        header = f"""# ======================================================================
# !! 警告: これはAI分析用の合成ファイルです !!
# ======================================================================
# このファイル (`COMBINED_CODE.py`) は、分析しやすいように複数のソースファイルを
# 1つに結合したものです。プロジェクトの実際の構造ではありません。
# 正確な構造は `PROJECT_INFO.md` を参照してください。
# 生成日時: {datetime.datetime.now().isoformat()}
# 結合ファイル数: {len(all_files)}
# --- 結合されたファイル一覧 ---
"""
        combined_lines = [header.strip()]
        for i, (root, file_path) in enumerate(all_files):
            rel_path_str = f"{root.name}/{file_path.relative_to(root)}" if root in file_path.parents else file_path.name
            combined_lines.append(f"# {i+1:03d}: {rel_path_str}")
        combined_lines.append("# ======================================================================\n")
        for root, file_path in all_files:
            rel_path_str = f"{root.name}/{file_path.relative_to(root)}" if root in file_path.parents else file_path.name
            combined_lines.append(f"\n# --- START: {rel_path_str} ---\n")
            try: combined_lines.append(file_path.read_text("utf-8", "ignore"))
            except Exception as e: combined_lines.append(f"# FAILED to read {file_path}: {e}")
            combined_lines.append(f"\n# --- END: {rel_path_str} ---\n")
        (session_dir / "COMBINED_CODE.py").write_text("\n".join(combined_lines), "utf-8")
        files_to_archive.append("COMBINED_CODE.py")
    # 5. README
    readme_content = create_readme_md(roots, "PROJECT_CHRONICLE.md" in files_to_archive, "PROJECT_BLUEPRINT.md" in files_to_archive, export_mode)
    (session_dir / "README.md").write_text(readme_content, "utf-8")
    files_to_archive.append("README.md")
    return files_to_archive

def create_readme_md(roots: List[Path], has_chronicle: bool, has_blueprint: bool, export_mode: str) -> str:
    title = f"# AI Analysis Pack for {generate_project_prefix(roots)}"
    lines = [title, "This ZIP is optimized for analysis by AI models. It contains:"]
    reading_order = 1
    if has_chronicle:
        lines.append(f"\n{reading_order}.  **`PROJECT_CHRONICLE.md` (【推奨】ここから読む)**\n    * Git履歴から生成されたプロジェクトの進化の物語。開発の文脈を理解できます。")
        reading_order += 1
    if has_blueprint:
        lines.append(f"\n{reading_order}.  **`PROJECT_BLUEPRINT.md` (【推奨】次に読む)**\n    * モジュール間の依存関係を可視化した構造設計図。")
        reading_order += 1
    lines.append(f"\n{reading_order}.  **`PROJECT_INFO.md`**\n    * プロジェクトの統計情報と完全なディレクトリ構造。")
    reading_order += 1
    if export_mode == 'full_context':
        lines.append(f"\n{reading_order}.  **`source_code/` (【重要】分析対象)**\n    * **元のディレクトリ構造を維持したソースコード一式。** このフォルダを直接分析してください。")
    else: # gemini
        lines.append(f"\n{reading_order}.  **`COMBINED_CODE.py`**\n    * 全ソースコードを結合した合成ファイル。**実際の構造ではありません。**")
    reading_order += 1
    lines.append(f"\n{reading_order}.  **`README.md` (このファイル)**")
    return "\n".join(lines)

def create_project_info_content(file_count: int, all_files: List[Tuple[Path, Path]], roots: List[Path], profile: str, mode: str) -> str:
    total_lines = 0; stats = defaultdict(lambda: {"files": 0, "lines": 0})
    for _, p in all_files:
        ext = p.suffix or "other"; lines = loc_count(p)
        stats[ext]["files"] += 1; stats[ext]["lines"] += lines; total_lines += lines
    tree_parts = []
    for root in roots:
        root_files = [p for r, p in all_files if r == root]
        if not root_files: continue
        tree_dict = {}
        for f in root_files:
            node = tree_dict
            for part in f.relative_to(root).parts: node = node.setdefault(part, {})
        lines = [f"📁 **{root.name}/**"]
        def build_tree(d, prefix=""):
            items = sorted(d.keys())
            for i, name in enumerate(items):
                connector = "└── " if i == len(items) - 1 else "├── "
                lines.append(f"{prefix}{connector}{name}")
                if d[name]: build_tree(d[name], prefix + ("    " if i == len(items) - 1 else "│   "))
        build_tree(tree_dict); tree_parts.append("\n".join(lines))
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
{"".join(f"| `{ext}` | {data['files']:,} | {data['lines']:,} |\\n" for ext, data in sorted(stats.items(), key=lambda item: item[1]['files'], reverse=True))}
## 🌳 Directory Structure
```
{chr(10).join(tree_parts)}
```"""

# --- Main Logic ---
def run_export(roots_str: str, profile_name: str, redact_env: bool, progress=gr.Progress(track_tqdm=True)):
    stop_event.clear()
    roots_list = [p.strip() for p in roots_str.splitlines() if p.strip()]
    if not roots_list: return None, "⚠️ 少なくとも1つのフォルダを選択してください"
    roots = [Path(p).expanduser().resolve() for p in roots_list]
    if not all(r.is_dir() for r in roots): return None, "⚠️ 無効なフォルダパスが含まれています"
    profile = PROFILES[profile_name]
    keep_set, include_dirs, export_mode = set(profile["keep"]), set(profile["include_dirs"]), profile["export_mode"]

    size_map = {'gemini': MAX_AI_CODE_SIZE_MB, 'full_context': MAX_GPT5_CODE_SIZE_MB}
    max_size_bytes = size_map.get(export_mode, MAX_TOTAL_SIZE_MB) * 1024 * 1024

    try:
        all_scored_files, full_graph = [], nx.DiGraph() if nx else None
        for i, root in enumerate(roots):
            progress(i / len(roots) * 0.1, desc=f"プロジェクト解析中: {root.name}")
            class SubProgress:
                def __call__(self, value, desc=""): progress(i / len(roots) * 0.7 + value * 0.7 / len(roots), desc=desc)
            scored, graph = collect_and_score_files(root, SubProgress(), keep_set, include_dirs)
            all_scored_files.extend([(score, root, path) for score, path in scored])
            if nx and graph: full_graph.add_edges_from(graph.edges())
        all_scored_files.sort(key=lambda x: x[0], reverse=True)
        final_files, final_paths, current_size = [], set(), 0
        for _, root, path in all_scored_files:
            if path in final_paths: continue
            size = path.stat().st_size
            if current_size + size <= max_size_bytes:
                final_files.append((root, path)); final_paths.add(path); current_size += size
        if not final_files: return None, "⚠️ 対象ファイルが見つかりませんでした"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = EXPORT_ROOT / f"{generate_project_prefix(roots)}_{profile_name.replace(' ', '')}_{ts}"
        session_dir.mkdir(exist_ok=True)
        if redact_env:
            for root in roots:
                if (root / ".env").exists(): write_redacted_env(root / ".env", session_dir)

        if export_mode in ['gemini', 'full_context']:
            export_ai_pack(session_dir, final_files, roots, profile_name, progress, full_graph, export_mode)
        else:
            export_standard(session_dir, final_files, roots, profile_name, progress)

        progress(0.95, desc="ZIPアーカイブを作成中...")
        zip_path = session_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in session_dir.rglob("*"):
                if f.is_file(): zf.write(f, f.relative_to(session_dir))

        mb = zip_path.stat().st_size / (1024 * 1024)
        return str(zip_path), f"✅ エクスポート完了 ({profile_name})\n- サイズ: {mb:.2f} MB\n- 出力先: {zip_path}"
    except Exception as e:
        return None, f"❌ エラー:\n{e}\n\n詳細:\n{traceback.format_exc(limit=3)}"

def cancel_export():
    stop_event.set()
    return "⏹️ キャンセル信号を送信しました"

# --- Gradio UI ---
def create_interface():
    # Gradio 4: Soft theme without primary_hue kw (removed)
    theme = getattr(gr.themes, "Soft", lambda: None)()
    with gr.Blocks(title="Atelier-Kyo Code Analyzer", theme=theme, analytics_enabled=False) as demo:
        gr.Markdown("# 🎨 Atelier-Kyo Code Analyzer (v5: GPT-5対応)")
        gr.Markdown("プロジェクトの目的に応じて、最適な形でコードを分析・パッケージ化します。**GPT-5 Full Context Pack**では、AIがプロジェクトをありのままに理解するための**構造化ソースコード**を提供します。")
        with gr.Row():
            dirs_tb = gr.Textbox(label="📁 プロジェクトフォルダ", value=DEFAULT_PROJECT_PATH, lines=2)
            profile_dd = gr.Dropdown(label="🎯 分析プロファイル", choices=list(PROFILES.keys()), value=DEFAULT_PROFILE)
        profile_info = gr.Markdown(f"**説明:** {PROFILES[DEFAULT_PROFILE]['description']}")
        def update_profile_info(profile_name): return gr.update(value=f"**説明:** {PROFILES[profile_name]['description']}")
        profile_dd.change(update_profile_info, inputs=profile_dd, outputs=profile_info)
        redact_ck = gr.Checkbox(label="🔑 .env ファイルをマスクして同梱 (.env.redacted)", value=True)
        with gr.Row():
            exp_btn = gr.Button("🚀 エクスポート開始", variant="primary", size="lg")
        cancel_btn = gr.Button("⏹️ キャンセル")
        zip_out = gr.File(label="📦 ダウンロード")
        status = gr.Textbox(label="📋 ステータス", lines=5, show_copy_button=True)
        export_event = exp_btn.click(fn=run_export, inputs=[dirs_tb, profile_dd, redact_ck], outputs=[zip_out, status], show_progress="full")
        cancel_btn.click(fn=cancel_export, cancels=[export_event], outputs=[status])
    return demo

if __name__ == "__main__":
    print("🎨 Atelier-Kyo Code Analyzer (v5) 起動中...")
    print(f"📁 エクスポート先: {EXPORT_ROOT}")
    allowed_paths = [str(EXPORT_ROOT), DEFAULT_PROJECT_PATH]
    app = create_interface()
    # ポートは環境変数 GRADIO_SERVER_PORT を優先。未指定なら自動割当（0）
    _port_raw = os.getenv("GRADIO_SERVER_PORT")
    try:
        server_port = int(_port_raw) if _port_raw else None
    except ValueError:
        server_port = None
    # Gradio 4: avoid forced browser open, allow share link for environments without localhost access
    app.launch(
        inbrowser=False,
        share=True,
        server_name="127.0.0.1",
        server_port=server_port,
        allowed_paths=allowed_paths,
        show_api=False,
    )
