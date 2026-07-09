"""BrowserOrchestrator: PLP/PDP flow orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agents.healing.failure_analysis_agent import FailureAnalysisAgent
from app.agents.healing.selector_repair_agent import SelectorRepairAgent
from app.agents.healing.self_healing_patch_agent import SelfHealingPatchAgent
from app.agents.healing.self_healing_patch_applier import SelfHealingPatchApplier
from app.agents.healing.self_healing_sandbox import SelfHealingSandbox
from app.agents.selector_discovery_agent import SelectorDiscoveryAgent

from .config_and_metrics import ConfigAndMetricsMixin
from .plp_pdp_flow import PlpPdpFlowMixin
from .self_healing import SelfHealingMixin
from .success_stage import SuccessStageMixin

try:
    from app.agents.healing.self_healing_policy import SelfHealingPolicy
except ImportError:
    SelfHealingPolicy = None  # type: ignore

logger = logging.getLogger(__name__)


class BrowserOrchestrator(
    PlpPdpFlowMixin,
    SuccessStageMixin,
    SelfHealingMixin,
    ConfigAndMetricsMixin,
):
    """
    BrowserUseAgent と NavigationDriver/PlpDriver/SelectorDiscoveryAgent の間に立ち、
    PLP→PDP フロー全体の状態遷移とエラー処理を一元管理するオーケストレータ。
    """

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any] | None = None,
        analysis_agent: Any | None = None,
        discovery_agent: Any | None = None,
        patch_agent: Any | None = None,
        sandbox: Any | None = None,
        policy: Any | None = None,
        patch_applier: Any | None = None,
        selector_repair_agent: Any | None = None,
        llm_client: Any | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.runtime_kwargs: dict[str, Any] = runtime_kwargs or {}
        self.log = log or logger

        self.analysis_agent = analysis_agent or FailureAnalysisAgent(runtime_kwargs=self.runtime_kwargs)
        self.discovery_agent = discovery_agent or SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
        self.patch_agent = patch_agent or SelfHealingPatchAgent(runtime_kwargs=self.runtime_kwargs)
        self.sandbox = sandbox or SelfHealingSandbox()
        self.policy = policy or (
            SelfHealingPolicy.from_file(Path("app/config/self_healing_policy.json")) if SelfHealingPolicy else None
        )
        self.patch_applier = patch_applier or SelfHealingPatchApplier()
        self.selector_repair_agent = selector_repair_agent or SelectorRepairAgent(llm_client=llm_client)

        self._overrides_path = Path("app/config/sites/overrides.local.json")
