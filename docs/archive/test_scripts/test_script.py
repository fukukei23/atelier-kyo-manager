#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')

import os
from dotenv import load_dotenv
load_dotenv('/home/yn441611/atelier-kyo-manager/.env')

from openai import OpenAI

glm_key = os.getenv("GLM_API_KEY")

client = OpenAI(api_key=glm_key, base_url="https://api.z.ai/api/coding/paas/v4")

response = client.chat.completions.create(
    model="glm-5",
    messages=[{"role": "user", "content": "Say 'OK'"}],
    max_tokens=50
)

msg = response.choices[0].message
content = msg.content or msg.reasoning_content or ""
print(f"Content: '{content}'")

# Test ai_llm_controller
print()
print("=== Testing AILlmController ===")
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')
# Mock opentelemetry
sys.modules['opentelemetry'] = type(sys)('opentelemetry')
sys.modules['opentelemetry.trace'] = type(sys)('opentelemetry.trace')
sys.modules['opentelemetry.trace'].get_tracer = lambda x: type('Tracer', (), {'start_as_current_span': lambda s, n: type('Span', (), {'__enter__': lambda s: s, '__exit__': lambda s, *a: None, 'set_attribute': lambda s, k, v: None})()})()

from app.utils.ai_llm_controller import AILlmController
c = AILlmController()
print(f"glm_client: {c.glm_client}")
if c.glm_client:
    result = c.generate("Say 'OK' in 5 chars", task_type="default")
    print(f"generate() result: '{result.text}'")
