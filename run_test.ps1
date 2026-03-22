cd C:\Users\USER\tools\atelier-kyo-manager
& ".venv\Scripts\python.exe" -c "from app.utils.ai_llm_controller import AILlmController; c = AILlmController(); print('minimax:', c.minimax_client); print('glm:', c.glm_client)"
