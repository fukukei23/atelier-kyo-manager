# ----------------------------------------------------------------
#  ファイル名: ai_llm_controller.py (製品版)
#  役割:
#    複数のLLMエンジンを透過的に呼び出すための統合コントローラー。
# ----------------------------------------------------------------

import google.generativeai as genai
from openai import OpenAI
from flask import current_app
import logging

class AILlmController:
    """複数のLLMエンジンを透過的に呼び出すための統合コントローラー。"""
    def __init__(self):
        # --- Geminiの初期化 ---
        try:
            self.gemini_api_key = current_app.config.get('GEMINI_API_KEY')
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
                logging.info("Gemini client initialized successfully.")
            else:
                self.gemini_model = None
                logging.warning("Gemini API key not found. Gemini will be unavailable.")
        except Exception as e:
            self.gemini_model = None
            logging.error(f"Failed to initialize Gemini: {e}")

        # --- DeepSeek (OpenAI互換)の初期化 ---
        try:
            self.deepseek_api_key = current_app.config.get('DEEPSEEK_API_KEY')
            self.deepseek_api_base = current_app.config.get('DEEPSEEK_API_BASE')
            if self.deepseek_api_key and self.deepseek_api_base:
                self.deepseek_client = OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url=self.deepseek_api_base
                )
                logging.info("DeepSeek client initialized successfully.")
            else:
                self.deepseek_client = None
                logging.warning("DeepSeek API key or base URL not found. DeepSeek will be unavailable.")
        except Exception as e:
            self.deepseek_client = None
            logging.error(f"Failed to initialize DeepSeek client: {e}")

    def generate(self, prompt: str, model_name: str = 'gemini') -> str:
        """指定されたモデルでテキストを生成する。"""
        logging.info(f"Generating text with model: {model_name}")
        try:
            if model_name.lower() == 'gemini':
                return self._generate_with_gemini(prompt)
            elif model_name.lower() == 'deepseek':
                return self._generate_with_deepseek(prompt)
            else:
                raise ValueError(f"Unsupported model: {model_name}. Please use 'gemini' or 'deepseek'.")
        except Exception as e:
            logging.error(f"Error generating text with {model_name}: {e}", exc_info=True)
            return f"エラー: テキスト生成に失敗しました ({model_name})"

    def _generate_with_gemini(self, prompt: str) -> str:
        if not self.gemini_model:
            raise ConnectionError("Gemini client is not initialized.")
        response = self.gemini_model.generate_content(prompt)
        return response.text

    def _generate_with_deepseek(self, prompt: str) -> str:
        if not hasattr(self, 'deepseek_client') or not self.deepseek_client:
            raise ConnectionError("DeepSeek client is not initialized.")
        chat_completion = self.deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
