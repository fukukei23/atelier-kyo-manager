#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')

print("インポートテスト...")

try:
    from app.agents.browser_use_agent import BrowserUseAgent
    print("BrowserUseAgent import OK")
except Exception as e:
    print(f"BrowserUseAgent import ERROR: {e}")

try:
    from app.agents.self_healing_agent import SelfHealingAgent
    print("SelfHealingAgent import OK")
except Exception as e:
    print(f"SelfHealingAgent import ERROR: {e}")

try:
    from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
    print("SelectorDiscoveryAgent import OK")
except Exception as e:
    print(f"SelectorDiscoveryAgent import ERROR: {e}")
