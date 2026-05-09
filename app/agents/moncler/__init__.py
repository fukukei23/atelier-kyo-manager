"""
Moncler 専用モジュール群

CR-ATELIER-003 Phase B: Moncler 固有ロジックの BrowserUseAgent からの完全分離

このモジュールは、Moncler サイト専用の処理を集約します：
- moncler_plp_handler.py: PLP 抽出フロー担当
- moncler_pdp_handler.py: PDP 抽出／検証担当
- moncler_navigation_policy.py: リダイレクト・ロケール制御など
"""

from app.agents.moncler.moncler_navigation_policy import MonclerNavigationPolicy
from app.agents.moncler.moncler_pdp_handler import MonclerPdpHandler
from app.agents.moncler.moncler_plp_handler import MonclerPlpHandler

__all__ = [
    "MonclerPlpHandler",
    "MonclerPdpHandler",
    "MonclerNavigationPolicy",
]
