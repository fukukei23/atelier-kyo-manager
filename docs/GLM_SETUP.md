# GLM API 設定ガイド

## 概要

LLMをGLM（智譜AI）に統一しました。

## 1. APIキー取得

1. [智譜AI Platform](https://open.bigmodel.cn/)にアクセス
2. アカウント登録/ログイン
3. API Keysからキーを生成

## 2. .envファイル設定

```bash
GLM_API_KEY=your-actual-glm-api-key-here
```

## 3. 設定内容

|.env変数|値|説明|
|---|---|---|
|`GLM_API_KEY`|APIキー|認証用|
|`GLM_MODEL`|glm-4|使用モデル|
|`GLM_API_BASE`|https://open.bigmodel.cn/api/paas/v4|エンドポイント|

## 4. コスト

|glm-4|価格|
|---|---|
|Input|$0.10 / 1M tokens|
|Output|$0.50 / 1M tokens|

## 5. 動作確認

```bash
python -c "from app.utils.ai_llm_controller import AILlmController; c = AILlmController(); print('glm_client:', c.glm_client)"
```

`glm_client: None`の場合はAPIキーが未設定または無効です。
