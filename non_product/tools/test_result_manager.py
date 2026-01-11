#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テスト結果管理モジュール

テスト実行結果を構造化して保存し、レポートを自動生成します。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = None  # 後で初期化


@dataclass
class TestResult:
    """個別のテスト結果"""
    name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float = 0.0
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class TestSessionResult:
    """テストセッション全体の結果"""
    timestamp: str
    command: str
    return_code: int
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    tests: List[TestResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        data = asdict(self)
        data["tests"] = [asdict(t) for t in self.tests]
        return data


class TestResultParser:
    """pytest の出力を解析して TestSessionResult に変換"""
    
    def __init__(self):
        self.result = TestSessionResult(
            timestamp=datetime.now().isoformat(),
            command="",
            return_code=0,
        )
    
    def parse_pytest_output(self, stdout: str, stderr: str, return_code: int) -> TestSessionResult:
        """pytest の出力を解析"""
        self.result.stdout = stdout
        self.result.stderr = stderr
        self.result.return_code = return_code
        
        # サマリー行を解析
        self._parse_summary(stdout)
        
        # テスト結果を解析
        self._parse_tests(stdout)
        
        return self.result
    
    def _parse_summary(self, output: str) -> None:
        """サマリー行を解析"""
        # pytest の出力形式に対応
        # "=== 10 passed in 2.34s ==="
        # "10 passed, 2 failed, 1 skipped in 3.45s"
        # "FAILED tests/test_example.py::test_function - AssertionError"
        summary_patterns = [
            r"=== (\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) error)? in ([\d.]+)s ===",
            r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) error)? in ([\d.]+)s",
            r"(\d+) passed",
            r"(\d+) failed",
            r"(\d+) skipped",
            r"(\d+) error",
            r"in ([\d.]+)s",
        ]
        
        # 総テスト数を先に取得
        total_match = re.search(r"(\d+)\s+(?:test|tests)", output, re.IGNORECASE)
        if total_match:
            self.result.total_tests = int(total_match.group(1))
        
        # 各ステータスを個別に取得
        passed_match = re.search(r"(\d+)\s+passed", output, re.IGNORECASE)
        if passed_match:
            self.result.passed = int(passed_match.group(1))
        
        failed_match = re.search(r"(\d+)\s+failed", output, re.IGNORECASE)
        if failed_match:
            self.result.failed = int(failed_match.group(1))
        
        skipped_match = re.search(r"(\d+)\s+skipped", output, re.IGNORECASE)
        if skipped_match:
            self.result.skipped = int(skipped_match.group(1))
        
        error_match = re.search(r"(\d+)\s+error", output, re.IGNORECASE)
        if error_match:
            self.result.errors = int(error_match.group(1))
        
        # 実行時間を取得
        duration_match = re.search(r"in\s+([\d.]+)s", output)
        if duration_match:
            self.result.duration = float(duration_match.group(1))
        
        # 総テスト数が取得できなかった場合は計算
        if self.result.total_tests == 0:
            self.result.total_tests = (
                self.result.passed + 
                self.result.failed + 
                self.result.skipped + 
                self.result.errors
            )
        
        # サマリー情報を辞書に保存
        self.result.summary = {
            "total": self.result.total_tests,
            "passed": self.result.passed,
            "failed": self.result.failed,
            "skipped": self.result.skipped,
            "errors": self.result.errors,
            "duration": self.result.duration,
            "success_rate": (
                (self.result.passed / self.result.total_tests * 100)
                if self.result.total_tests > 0 else 0.0
            ),
        }
    
    def _parse_tests(self, output: str) -> None:
        """個別のテスト結果を解析"""
        lines = output.split("\n")
        current_test: Optional[TestResult] = None
        
        for i, line in enumerate(lines):
            # テスト開始行を検出
            # "tests/test_example.py::test_function PASSED [ 50%]"
            test_match = re.match(
                r"^tests?/.*?::(.*?)\s+(PASSED|FAILED|SKIPPED|ERROR)\s*",
                line
            )
            
            if test_match:
                # 前のテストを保存
                if current_test:
                    self.result.tests.append(current_test)
                
                test_name = test_match.group(1)
                status = test_match.group(2).lower()
                
                current_test = TestResult(
                    name=test_name,
                    status=status,
                )
                
                # ファイルパスと行番号を抽出
                file_match = re.match(r"^tests?/(.*?)::", line)
                if file_match:
                    current_test.file_path = f"tests/{file_match.group(1)}"
                
                # 実行時間を抽出
                duration_match = re.search(r"\[[\s\d]+%\]\s*\[?\s*([\d.]+)s\]?", line)
                if duration_match:
                    current_test.duration = float(duration_match.group(1))
            
            # 失敗時のエラーメッセージを取得
            elif current_test and current_test.status == "failed":
                if "FAILED" in line or "AssertionError" in line or "Error" in line:
                    if not current_test.error_message:
                        current_test.error_message = line.strip()
                elif current_test.error_message and not current_test.error_traceback:
                    # トレースバックの開始
                    if "Traceback" in line or "  File" in line:
                        current_test.error_traceback = line.strip()
                elif current_test.error_traceback:
                    # トレースバックの続き
                    if line.strip():
                        current_test.error_traceback += "\n" + line.strip()
                    else:
                        # 空行で終了
                        pass
        
        # 最後のテストを保存
        if current_test:
            self.result.tests.append(current_test)


class TestResultManager:
    """テスト結果の管理とレポート生成"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "docs" / "test_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, session_result: TestSessionResult, command: str) -> Path:
        """テスト結果をJSON形式で保存"""
        session_result.command = command
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.results_dir / f"test_result_{timestamp}.json"
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(session_result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return json_file
    
    def generate_markdown_report(self, session_result: TestSessionResult, json_file: Path) -> Path:
        """Markdown形式のレポートを生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"test_report_{timestamp}.md"
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(self._format_markdown_report(session_result, json_file))
        
        return report_file
    
    def _format_markdown_report(self, session_result: TestSessionResult, json_file: Path) -> str:
        """Markdownレポートをフォーマット"""
        lines = []
        lines.append("# テスト実行レポート")
        lines.append("")
        lines.append(f"**実行日時**: {session_result.timestamp}")
        lines.append(f"**コマンド**: `{session_result.command}`")
        lines.append(f"**終了コード**: {session_result.return_code}")
        lines.append("")
        
        # サマリー
        lines.append("## サマリー")
        lines.append("")
        lines.append(f"- **総テスト数**: {session_result.summary.get('total', 0)}")
        lines.append(f"- **成功**: {session_result.summary.get('passed', 0)} ✅")
        lines.append(f"- **失敗**: {session_result.summary.get('failed', 0)} ❌")
        lines.append(f"- **スキップ**: {session_result.summary.get('skipped', 0)} ⏭️")
        lines.append(f"- **エラー**: {session_result.summary.get('errors', 0)} ⚠️")
        lines.append(f"- **実行時間**: {session_result.summary.get('duration', 0.0):.2f}秒")
        lines.append(f"- **成功率**: {session_result.summary.get('success_rate', 0.0):.1f}%")
        lines.append("")
        
        # ステータス判定
        if session_result.return_code == 0:
            lines.append("## ✅ すべてのテストが成功しました")
        else:
            lines.append("## ❌ テストが失敗しました")
        lines.append("")
        
        # 失敗したテストの詳細
        failed_tests = [t for t in session_result.tests if t.status == "failed"]
        if failed_tests:
            lines.append("## 失敗したテスト")
            lines.append("")
            for test in failed_tests:
                lines.append(f"### {test.name}")
                lines.append("")
                if test.file_path:
                    lines.append(f"**ファイル**: `{test.file_path}`")
                    if test.line_number:
                        lines.append(f"**行番号**: {test.line_number}")
                lines.append("")
                if test.error_message:
                    lines.append("**エラーメッセージ**:")
                    lines.append("```")
                    lines.append(test.error_message)
                    lines.append("```")
                    lines.append("")
                if test.error_traceback:
                    lines.append("**トレースバック**:")
                    lines.append("```")
                    lines.append(test.error_traceback)
                    lines.append("```")
                    lines.append("")
        
        # すべてのテスト一覧
        if session_result.tests:
            lines.append("## テスト一覧")
            lines.append("")
            lines.append("| テスト名 | ステータス | 実行時間 |")
            lines.append("|---------|-----------|---------|")
            
            for test in session_result.tests:
                status_icon = {
                    "passed": "✅",
                    "failed": "❌",
                    "skipped": "⏭️",
                    "error": "⚠️",
                }.get(test.status, "❓")
                
                lines.append(
                    f"| {test.name} | {status_icon} {test.status} | {test.duration:.3f}s |"
                )
            lines.append("")
        
        # 詳細情報へのリンク
        lines.append("## 詳細情報")
        lines.append("")
        lines.append(f"- **JSON結果ファイル**: `{json_file.relative_to(self.project_root)}`")
        lines.append("")
        
        return "\n".join(lines)


def run_tests_with_report(
    test_path: str = "tests/",
    extra_args: Optional[List[str]] = None,
    python_executable: Optional[str] = None,
) -> tuple[TestSessionResult, Path, Path]:
    """
    テストを実行して結果を保存・レポート生成
    
    Returns:
        (TestSessionResult, json_file_path, markdown_report_path)
    """
    project_root = Path(__file__).parent.parent
    
    # Python実行可能ファイルを決定
    if python_executable is None:
        python_executable = sys.executable
    
    # コマンドを構築
    cmd = [python_executable, "-m", "pytest", test_path, "-v", "--tb=short"]
    if extra_args:
        cmd.extend(extra_args)
    
    command_str = " ".join(cmd)
    
    print("=" * 80)
    print("テスト実行")
    print("=" * 80)
    print(f"コマンド: {command_str}")
    print()
    
    # テスト実行
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,  # 10分タイムアウト
        )
    except subprocess.TimeoutExpired:
        result = subprocess.CompletedProcess(
            cmd,
            returncode=1,
            stdout="",
            stderr="テストがタイムアウトしました（10分）",
        )
    except Exception as e:
        result = subprocess.CompletedProcess(
            cmd,
            returncode=1,
            stdout="",
            stderr=f"テスト実行エラー: {e}",
        )
    
    # 結果を解析
    parser = TestResultParser()
    session_result = parser.parse_pytest_output(
        result.stdout,
        result.stderr,
        result.returncode,
    )
    
    # 結果を保存
    manager = TestResultManager(project_root)
    json_file = manager.save_result(session_result, command_str)
    report_file = manager.generate_markdown_report(session_result, json_file)
    
    print("=" * 80)
    print("テスト実行完了")
    print("=" * 80)
    print(f"JSON結果: {json_file.relative_to(project_root)}")
    print(f"レポート: {report_file.relative_to(project_root)}")
    print("=" * 80)
    
    return session_result, json_file, report_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="テストを実行して結果をレポート生成")
    parser.add_argument(
        "test_path",
        nargs="?",
        default="tests/",
        help="テストパス（デフォルト: tests/）",
    )
    parser.add_argument(
        "--args",
        nargs="*",
        help="追加のpytest引数",
    )
    
    args = parser.parse_args()
    
    session_result, json_file, report_file = run_tests_with_report(
        test_path=args.test_path,
        extra_args=args.args,
    )
    
    # 終了コードで終了
    sys.exit(session_result.return_code)

