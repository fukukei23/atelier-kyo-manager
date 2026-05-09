# ======================================================================
# File: app/utils/pdf_tool.py
# Registry: app/utils/pdf_tool.py
# Rev: 1.1 (2025-09-10 15:02 JST)
#
# 目的:
# - HTMLファイルをPDFに変換する薄いユーティリティ。
# - WeasyPrint を優先使用。未インストール時は wkhtmltopdf を試行。
#
# 使用例:
# from app.utils.pdf_tool import convert_html_file_to_pdf
# pdf_path = convert_html_file_to_pdf("output/reports/daily_report_GUCCI_20250910.html")
# ======================================================================
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _weasyprint_available() -> bool:
    """Check if weasyprint is installed and available."""
    try:
        import weasyprint  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _wkhtmltopdf_available() -> str | None:
    """Check if wkhtmltopdf executable is in PATH."""
    return shutil.which("wkhtmltopdf")


def convert_html_file_to_pdf(html_path: str | Path, out_pdf: str | Path | None = None, engine: str = "auto") -> Path:
    """
    Converts an HTML file to a PDF document.

    Args:
        html_path: Path to the source HTML file.
        out_pdf: Optional path for the output PDF. If None, defaults to the same name as the HTML file.
        engine: The conversion engine to use. Can be "auto", "weasyprint", or "wkhtmltopdf".

    Returns:
        The path to the generated PDF file.

    Raises:
        FileNotFoundError: If the source HTML file does not exist.
        RuntimeError: If no suitable PDF conversion engine is found.
        subprocess.CalledProcessError: If wkhtmltopdf fails.
    """
    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML source file not found: {html_path}")

    if out_pdf is None:
        out_pdf = html_path.with_suffix(".pdf")
    out_pdf = Path(out_pdf).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # 1) Try WeasyPrint engine
    if engine in ("auto", "weasyprint") and _weasyprint_available():
        logging.info(f"Using 'weasyprint' engine to convert {html_path.name}")
        try:
            from weasyprint import HTML

            HTML(filename=str(html_path)).write_pdf(str(out_pdf))
            logging.info(f"Successfully created PDF: {out_pdf}")
            return out_pdf
        except Exception as e:
            logging.error(f"WeasyPrint failed: {e}")
            if engine == "weasyprint":
                raise  # Re-raise if WeasyPrint was explicitly requested

    # 2) Fallback to wkhtmltopdf engine
    wk_executable = _wkhtmltopdf_available()
    if (engine in ("auto", "wkhtmltopdf")) and wk_executable:
        logging.info(f"Using 'wkhtmltopdf' engine to convert {html_path.name}")
        try:
            # --quiet suppresses console output for cleaner logs
            command = [wk_executable, "--quiet", str(html_path), str(out_pdf)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Successfully created PDF: {out_pdf}")
            return out_pdf
        except subprocess.CalledProcessError as e:
            logging.error(f"wkhtmltopdf failed with exit code {e.returncode}.")
            logging.error(f"Stderr: {e.stderr}")
            raise  # Re-raise the exception

    # If we reach here, no engine was found or available
    raise RuntimeError(
        "No PDF engine available or the selected engine failed. "
        "Please install 'weasyprint' (`pip install weasyprint`) or 'wkhtmltopdf'."
    )
