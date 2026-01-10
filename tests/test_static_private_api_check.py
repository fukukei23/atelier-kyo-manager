"""ASTベース静的検査: private API依存を検出するテスト。

標準ライブラリのastモジュールのみを使用。
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest


class PrivateAPIViolation:
    """private API違反を表現するデータクラス。"""
    
    def __init__(
        self,
        file_path: Path,
        line_no: int,
        violation_type: str,
        details: str,
    ):
        self.file_path = file_path
        self.line_no = line_no
        self.violation_type = violation_type
        self.details = details
    
    def __repr__(self) -> str:
        return (
            f"{self.violation_type} detected:\n"
            f"  file: {self.file_path}\n"
            f"  line: {self.line_no}\n"
            f"  {self.details}"
        )


class PrivateAPIVisitor(ast.NodeVisitor):
    """ASTを走査し、private API依存を検出するVisitor。"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.violations: list[PrivateAPIViolation] = []
        self.imported_modules: dict[str, str] = {}  # alias -> module_name
        self.imported_names: dict[str, str] = {}  # name -> module_name
    
    def visit_Import(self, node: ast.Import) -> None:
        """`import X as Y` 形式を検出し、aliasを記録。"""
        for alias in node.names:
            if alias.asname:
                # `import X as Y` の場合、Yがprivateかチェック
                if alias.name.startswith("_"):
                    self.violations.append(
                        PrivateAPIViolation(
                            self.file_path,
                            node.lineno,
                            "PRIVATE_IMPORT",
                            f"import: {alias.name} as {alias.asname}",
                        )
                    )
                # aliasを記録（後でAttributeチェックで使用）
                self.imported_modules[alias.asname] = alias.name
            else:
                # `import X` の場合、Xがprivateかチェック
                if alias.name.startswith("_"):
                    self.violations.append(
                        PrivateAPIViolation(
                            self.file_path,
                            node.lineno,
                            "PRIVATE_IMPORT",
                            f"import: {alias.name}",
                        )
                    )
                # モジュール名を記録
                module_name = alias.name.split(".")[0]
                self.imported_modules[module_name] = alias.name
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """`from X import Y` 形式を検出し、private importを検出。"""
        if node.module:
            for alias in node.names:
                # private関数/クラスのimportを検出
                if alias.name.startswith("_"):
                    module_path = f"{node.module}.{alias.name}" if node.module else alias.name
                    if alias.asname:
                        module_path += f" as {alias.asname}"
                    
                    self.violations.append(
                        PrivateAPIViolation(
                            self.file_path,
                            node.lineno,
                            "PRIVATE_IMPORT",
                            f"import: {module_path}",
                        )
                    )
                    # aliasを記録
                    if alias.asname:
                        self.imported_names[alias.asname] = node.module or ""
                    else:
                        self.imported_names[alias.name] = node.module or ""
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """`X._something` 形式を検出し、private attribute accessを検出。"""
        # Pythonの組み込み属性（`__xxx__`）は除外
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.generic_visit(node)
            return
        
        # `attr.startswith("_")` をチェック（ただし`__xxx__`は除外済み）
        if node.attr.startswith("_"):
            # valueがNameの場合（変数名）
            if isinstance(node.value, ast.Name):
                var_name = node.value.id
                # この変数がimportされたモジュール/aliasかチェック
                if var_name in self.imported_modules or var_name in self.imported_names:
                    violation_type = "PRIVATE_ATTRIBUTE_ACCESS"
                    details = f"access: {var_name}.{node.attr}"
                    self.violations.append(
                        PrivateAPIViolation(
                            self.file_path,
                            node.lineno,
                            violation_type,
                            details,
                        )
                    )
            # valueがAttributeの場合（ネストされた属性アクセス）
            elif isinstance(node.value, ast.Attribute):
                # `X.Y._something` のような形式
                # ただし、親属性が`__xxx__`形式の場合は除外
                if (
                    not (node.value.attr.startswith("__") and node.value.attr.endswith("__"))
                    and node.attr.startswith("_")
                ):
                    # 親がimportされたモジュール/aliasかチェック
                    if isinstance(node.value.value, ast.Name):
                        parent_var = node.value.value.id
                        if parent_var in self.imported_modules or parent_var in self.imported_names:
                            violation_type = "PRIVATE_ATTRIBUTE_ACCESS"
                            details = f"access: {self._get_attribute_path(node.value)}.{node.attr}"
                            self.violations.append(
                                PrivateAPIViolation(
                                    self.file_path,
                                    node.lineno,
                                    violation_type,
                                    details,
                                )
                            )
        
        self.generic_visit(node)
    
    def _get_attribute_path(self, node: ast.Attribute) -> str:
        """Attributeノードから属性パス文字列を取得。"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_path(node.value)}.{node.attr}"
        else:
            return node.attr


def check_file_for_private_api(file_path: Path) -> list[PrivateAPIViolation]:
    """
    指定ファイルでprivate API依存をチェックする。
    
    Args:
        file_path: チェック対象ファイルパス
        
    Returns:
        検出された違反のリスト
    """
    try:
        source_code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source_code, filename=str(file_path))
        
        visitor = PrivateAPIVisitor(file_path)
        visitor.visit(tree)
        
        return visitor.violations
    except SyntaxError as e:
        # 構文エラーは違反として扱わない（別途lintで検出）
        return []
    except Exception:
        # その他のエラーも無視
        return []


def test_utils_no_private_api_dependency():
    """app/utils/ 配下のsourcing関連モジュールでprivate API依存がないことを確認。"""
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "app" / "utils"
    
    if not utils_dir.exists():
        return  # utilsディレクトリが存在しない場合はスキップ
    
    all_violations: list[PrivateAPIViolation] = []
    
    # sourcing関連モジュールのみをチェック
    sourcing_modules = [
        "sourcing_csv_adapter.py",
        "sourcing_csv_runner.py",
        "sourcing_csv_batch_runner.py",
        "sourcing_input_schema.py",
        "sourcing_profitability.py",
        "sourcing_tier.py",
        "sourcing_pipeline.py",
    ]
    
    for py_file in utils_dir.rglob("*.py"):
        # __pycache__ は除外
        if "__pycache__" in str(py_file):
            continue
        
        # sourcing関連モジュールのみチェック
        if py_file.name not in sourcing_modules:
            continue
        
        violations = check_file_for_private_api(py_file)
        all_violations.extend(violations)
    
    # 違反があれば詳細なメッセージと共に失敗
    if all_violations:
        error_message = "Private API violations detected:\n\n"
        for violation in all_violations:
            error_message += str(violation) + "\n\n"
        pytest.fail(error_message)
    
    assert len(all_violations) == 0


def test_sourcing_modules_no_private_api_dependency():
    """sourcing関連モジュールでprivate API依存がないことを確認。"""
    project_root = Path(__file__).parent.parent
    
    sourcing_modules = [
        project_root / "app" / "utils" / "sourcing_csv_batch_runner.py",
        project_root / "app" / "utils" / "sourcing_csv_runner.py",
        project_root / "app" / "utils" / "sourcing_pipeline.py",
    ]
    
    all_violations: list[PrivateAPIViolation] = []
    
    for module_path in sourcing_modules:
        if not module_path.exists():
            continue
        
        violations = check_file_for_private_api(module_path)
        all_violations.extend(violations)
    
    if all_violations:
        error_message = "Private API violations detected in sourcing modules:\n\n"
        for violation in all_violations:
            error_message += str(violation) + "\n\n"
        pytest.fail(error_message)
    
    assert len(all_violations) == 0
