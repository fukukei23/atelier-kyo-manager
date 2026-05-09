from __future__ import annotations

try:
    from app.agents.plugins.base import StrategyPlugin
    from app.agents.plugins.moncler_plp_v1 import MonclerPLPStrategy
except Exception:
    StrategyPlugin = None  # type: ignore
    MonclerPLPStrategy = None  # type: ignore

PLUGIN_REGISTRY: dict[str, StrategyPlugin] = {}

if StrategyPlugin and MonclerPLPStrategy:
    PLUGIN_REGISTRY["MONCLER_OFFICIAL"] = MonclerPLPStrategy()
