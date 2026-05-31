"""Tests for app/utils/sourcing_pipeline.py - Issue #89"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.utils.sourcing_pipeline import main, run_pipeline


# ── run_pipeline ─────────────────────────────────────────────────────────────

class TestRunPipeline:
    def test_file_not_found_writes_invalid(self, tmp_path):
        """存在しない入力ファイル → status=invalid, tier=D が出力される"""
        input_path = tmp_path / "nonexistent.json"
        profit_out = tmp_path / "profit.json"
        tier_out = tmp_path / "tier.json"

        run_pipeline(input_path, profit_out, tier_out)

        assert profit_out.exists()
        assert tier_out.exists()

        profit = json.loads(profit_out.read_text())
        tier = json.loads(tier_out.read_text())
        assert profit["status"] == "invalid"
        assert tier["tier"] == "D"
        assert "errors" in profit

    def test_invalid_json_writes_invalid(self, tmp_path):
        """無効な JSON ファイル → status=invalid, tier=D が出力される"""
        input_path = tmp_path / "input.json"
        input_path.write_text("{not valid json}", encoding="utf-8")
        profit_out = tmp_path / "profit.json"
        tier_out = tmp_path / "tier.json"

        run_pipeline(input_path, profit_out, tier_out)

        profit = json.loads(profit_out.read_text())
        tier = json.loads(tier_out.read_text())
        assert profit["status"] == "invalid"
        assert tier["tier"] == "D"

    def test_output_dirs_auto_created(self, tmp_path):
        """出力ディレクトリが存在しなくても自動作成される"""
        input_path = tmp_path / "nonexistent.json"
        profit_out = tmp_path / "sub1" / "deep" / "profit.json"
        tier_out = tmp_path / "sub2" / "tier.json"

        run_pipeline(input_path, profit_out, tier_out)

        assert profit_out.parent.exists()
        assert tier_out.parent.exists()

    def test_normal_execution_calls_pipeline_functions(self, tmp_path):
        """正常実行: バリデーション・利益計算・Tier判定が呼ばれる"""
        input_data = {"product_name": "テスト商品", "selling_price": 50000}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(input_data), encoding="utf-8")
        profit_out = tmp_path / "profit.json"
        tier_out = tmp_path / "tier.json"

        mock_profit_result = {"status": "valid", "profit": 10000, "profit_rate": 0.2}
        mock_tier_result = {"tier": "B", "reason": "利益率20%"}

        with patch("app.utils.sourcing_pipeline.validate_sourcing_input") as mock_val,              patch("app.utils.sourcing_pipeline.calculate_profitability") as mock_calc,              patch("app.utils.sourcing_pipeline.judge_tier") as mock_tier:

            mock_val.return_value = {"normalized": input_data}
            mock_calc.return_value = mock_profit_result
            mock_tier.return_value = mock_tier_result

            run_pipeline(input_path, profit_out, tier_out)

            mock_val.assert_called_once_with(input_data)
            mock_calc.assert_called_once()
            mock_tier.assert_called_once()

        assert profit_out.exists()
        assert tier_out.exists()

    def test_normal_writes_json_output(self, tmp_path):
        """正常実行: 結果が JSON ファイルに保存される"""
        input_data = {"product": "bag"}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(input_data), encoding="utf-8")
        profit_out = tmp_path / "profit.json"
        tier_out = tmp_path / "tier.json"

        mock_profit = {"status": "valid", "profit": 5000}
        mock_tier = {"tier": "A", "reason": "高利益率"}

        with patch("app.utils.sourcing_pipeline.validate_sourcing_input") as mv,              patch("app.utils.sourcing_pipeline.calculate_profitability", return_value=mock_profit),              patch("app.utils.sourcing_pipeline.judge_tier", return_value=mock_tier):

            mv.return_value = {"normalized": input_data}
            run_pipeline(input_path, profit_out, tier_out)

        profit_content = json.loads(profit_out.read_text())
        tier_content = json.loads(tier_out.read_text())
        assert profit_content["status"] == "valid"
        assert tier_content["tier"] == "A"

    def test_path_objects_accepted(self, tmp_path):
        """Path オブジェクトを渡しても動作する"""
        profit_out = tmp_path / "p.json"
        tier_out = tmp_path / "t.json"
        # FileNotFoundError のパスで動作確認
        run_pipeline(Path("/nonexistent/path.json"), profit_out, tier_out)
        assert profit_out.exists()

    def test_string_paths_accepted(self, tmp_path):
        """文字列パスを渡しても動作する"""
        profit_out = tmp_path / "p.json"
        tier_out = tmp_path / "t.json"
        run_pipeline(str(tmp_path / "missing.json"), str(profit_out), str(tier_out))
        assert profit_out.exists()


# ── main ──────────────────────────────────────────────────────────────────────

class TestMain:
    def test_main_calls_run_pipeline(self):
        """main() が run_pipeline を呼ぶ"""
        with patch("app.utils.sourcing_pipeline.run_pipeline") as mock_run:
            main()
            mock_run.assert_called_once()

    def test_main_passes_three_paths(self):
        """main() は 3 つのパスを run_pipeline に渡す"""
        with patch("app.utils.sourcing_pipeline.run_pipeline") as mock_run:
            main()
            args = mock_run.call_args[1] if mock_run.call_args[1] else mock_run.call_args[0]
            # 3 引数が渡されていること
            if isinstance(args, dict):
                assert len(args) == 3
            else:
                assert len(args) == 3
