from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AIVisionAgent:
    """Best-effort parcel inspection stub. Connect to a real Vision API later."""

    def __init__(self, run_context: Optional[Any] = None) -> None:
        self.run_context = run_context

    def inspect_parcel_images(
        self,
        *,
        parcel_id: str,
        image_urls: List[str],
    ) -> Dict[str, Any]:
        analysis_results = [self._analyze_image(url) for url in image_urls]
        issues = self._summarize_issues(analysis_results)
        return {
            "parcel_id": parcel_id,
            "image_count": len(image_urls),
            "issues": issues,
            "raw_analysis": analysis_results,
        }

    def _analyze_image(self, url: str) -> Dict[str, Any]:
        logger.debug("[AIVisionAgent] Placeholder analysis for %s", url)
        return {
            "image_url": url,
            "damage_detected": False,
            "brand_logo_visible": True,
            "notes": "Vision analysis not implemented yet.",
        }

    def _summarize_issues(self, results: List[Dict[str, Any]]) -> List[str]:
        issues: List[str] = []
        if any(result.get("damage_detected") for result in results):
            issues.append("Potential damage detected.")
        if results and not any(result.get("brand_logo_visible") for result in results):
            issues.append("Brand logo not visible in provided photos.")
        return issues
