# tools/llm_smoketest.py
import json, os
from app.utils.ai_llm_controller import LLM  # 実際のクラス名/パスに合わせて

def main():
    llm = LLM()  # あなたのコントローラの初期化
    print("[probe] starting...")
    info = llm.probe()  # 例: {'active':'openai','order':['openai','deepseek'],...} を返す想定
    print("[probe] ->", info)
    msg = llm.chat("返答は3語: システム稼働確認")  # ほんの短いプロンプト
    print("[chat ] ->", msg)

if __name__ == "__main__":
    main()
