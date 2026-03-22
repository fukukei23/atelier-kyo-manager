#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')

from app.utils.ai_llm_controller import AILlmController

c = AILlmController()
print("minimax_client:", c.minimax_client)
print("glm_client:", c.glm_client)
print("TASK_TO_MODEL_FAMILY:", c.__class__.__dict__.get('__init__') or "OK")
