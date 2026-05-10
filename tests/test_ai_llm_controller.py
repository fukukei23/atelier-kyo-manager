"""
ai_llm_controller のテスト — キャッシュ・コスト・リトライ・フォールバックの検証
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app.models.result_models import GenerateResult
from app.utils.ai_llm_controller import AILlmController, MODEL_COST, TASK_TO_MODEL_FAMILY


# ==============================
# _cache_key のテスト
# ==============================


class TestCacheKey:
    def test_deterministic(self):
        ctrl = object.__new__(AILlmController)
        k1 = ctrl._cache_key("openai", "hello", None)
        k2 = ctrl._cache_key("openai", "hello", None)
        assert k1 == k2

    def test_different_prompt_different_key(self):
        ctrl = object.__new__(AILlmController)
        k1 = ctrl._cache_key("openai", "hello", None)
        k2 = ctrl._cache_key("openai", "world", None)
        assert k1 != k2

    def test_different_family_different_key(self):
        ctrl = object.__new__(AILlmController)
        k1 = ctrl._cache_key("openai", "hello", None)
        k2 = ctrl._cache_key("deepseek", "hello", None)
        assert k1 != k2

    def test_tools_affect_key(self):
        ctrl = object.__new__(AILlmController)
        k1 = ctrl._cache_key("openai", "hello", None)
        k2 = ctrl._cache_key("openai", "hello", [{"name": "tool1"}])
        assert k1 != k2

    def test_is_sha256_hex(self):
        ctrl = object.__new__(AILlmController)
        key = ctrl._cache_key("gemini", "test", [])
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ==============================
# _calc_cost のテスト
# ==============================


class TestCalcCost:
    def test_openai_pricing(self):
        ctrl = object.__new__(AILlmController)
        usage = {"input": 1000, "output": 500}
        cost = ctrl._calc_cost("openai", usage)
        expected = 1000 * MODEL_COST["openai"]["input"] + 500 * MODEL_COST["openai"]["output"]
        assert abs(cost - expected) < 1e-10

    def test_deepseek_pricing(self):
        ctrl = object.__new__(AILlmController)
        usage = {"input": 2000, "output": 1000}
        cost = ctrl._calc_cost("deepseek", usage)
        expected = 2000 * MODEL_COST["deepseek"]["input"] + 1000 * MODEL_COST["deepseek"]["output"]
        assert abs(cost - expected) < 1e-10

    def test_local_is_free(self):
        ctrl = object.__new__(AILlmController)
        cost = ctrl._calc_cost("local", {"input": 99999, "output": 99999})
        assert cost == 0.0

    def test_prompt_tokens_key_format(self):
        ctrl = object.__new__(AILlmController)
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = ctrl._calc_cost("openai", usage)
        expected = 1000 * MODEL_COST["openai"]["input"] + 500 * MODEL_COST["openai"]["output"]
        assert abs(cost - expected) < 1e-10

    def test_missing_keys_default_zero(self):
        ctrl = object.__new__(AILlmController)
        cost = ctrl._calc_cost("openai", {})
        assert cost == 0.0

    def test_unknown_family_zero_cost(self):
        ctrl = object.__new__(AILlmController)
        cost = ctrl._calc_cost("nonexistent_model", {"input": 1000, "output": 500})
        assert cost == 0.0

    def test_zero_usage(self):
        ctrl = object.__new__(AILlmController)
        cost = ctrl._calc_cost("openai", {"input": 0, "output": 0})
        assert cost == 0.0


# ==============================
# _sentiment のテスト
# ==============================


class TestSentiment:
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    def test_returns_neutral_when_no_nlp(self):
        ctrl = object.__new__(AILlmController)
        assert ctrl._sentiment("I love this!") == "NEUTRAL"

    @patch("app.utils.ai_llm_controller.LOCAL_NLP")
    def test_positive(self, mock_nlp):
        mock_nlp.return_value = [{"label": "POSITIVE", "score": 0.99}]
        ctrl = object.__new__(AILlmController)
        assert ctrl._sentiment("Great product!") == "POSITIVE"

    @patch("app.utils.ai_llm_controller.LOCAL_NLP")
    def test_negative(self, mock_nlp):
        mock_nlp.return_value = [{"label": "NEGATIVE", "score": 0.95}]
        ctrl = object.__new__(AILlmController)
        assert ctrl._sentiment("Terrible quality") == "NEGATIVE"

    @patch("app.utils.ai_llm_controller.LOCAL_NLP")
    def test_nlp_failure_returns_neutral(self, mock_nlp):
        mock_nlp.side_effect = RuntimeError("model load failed")
        ctrl = object.__new__(AILlmController)
        assert ctrl._sentiment("Some text") == "NEUTRAL"

    @patch("app.utils.ai_llm_controller.LOCAL_NLP")
    def test_truncates_long_text(self, mock_nlp):
        mock_nlp.return_value = [{"label": "POSITIVE", "score": 0.9}]
        ctrl = object.__new__(AILlmController)
        ctrl._sentiment("x" * 1000)
        call_arg = mock_nlp.call_args[0][0]
        assert len(call_arg) <= 512


# ==============================
# _generate_with_retry のテスト
# ==============================


class TestGenerateWithRetry:
    def _make_ctrl(self):
        ctrl = object.__new__(AILlmController)
        ctrl._initialized = True
        return ctrl

    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA", None)
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    @patch.object(AILlmController, "_raw_call")
    def test_success_on_first_attempt(self, mock_raw):
        mock_raw.return_value = ("Hello!", {"input": 100, "output": 50})
        ctrl = self._make_ctrl()

        result = ctrl._generate_with_retry("openai", "test", None, False, None)

        assert result.text == "Hello!"
        assert result.model_family == "openai"
        mock_raw.assert_called_once()

    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA", None)
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    @patch.object(AILlmController, "_raw_call")
    @patch("app.utils.ai_llm_controller.time")
    def test_fallback_from_gemini_to_openai(self, mock_time, mock_raw):
        mock_raw.side_effect = [
            RuntimeError("gemini down"),
            ("Fallback response", {"input": 50, "output": 30}),
        ]
        ctrl = self._make_ctrl()

        result = ctrl._generate_with_retry("gemini", "test", None, False, None)

        assert result.text == "Fallback response"
        assert result.model_family == "openai"
        assert mock_raw.call_count == 2

    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA", None)
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    @patch.object(AILlmController, "_raw_call")
    @patch("app.utils.ai_llm_controller.time")
    def test_all_fail_returns_none_instead_of_raising(self, mock_time, mock_raw):
        """local なしの場合、families=4 だが3回しか実行されない。
        attempt(3) != len(families)(4) なので raise されず None が返る"""
        mock_raw.side_effect = [RuntimeError("down")] * 10
        ctrl = self._make_ctrl()

        result = ctrl._generate_with_retry("gemini", "test", None, False, None)
        assert result is None

    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA", None)
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    @patch.object(AILlmController, "_stream_call")
    def test_stream_call_used_when_stream_true(self, mock_stream):
        mock_stream.return_value = ("Streamed text", {"input": 100, "output": 50})
        ctrl = self._make_ctrl()

        result = ctrl._generate_with_retry("openai", "test", None, True, lambda x: None)

        assert result.text == "Streamed text"
        mock_stream.assert_called_once()

    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA", None)
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    @patch.object(AILlmController, "_raw_call")
    @patch("app.utils.ai_llm_controller.time")
    def test_exponential_backoff(self, mock_time, mock_raw):
        mock_raw.side_effect = [
            RuntimeError("fail 1"),
            ("ok", {"input": 10, "output": 5}),
        ]
        ctrl = self._make_ctrl()

        ctrl._generate_with_retry("openai", "test", None, False, None)

        mock_time.sleep.assert_called_once_with(1.5)


# ==============================
# TASK_TO_MODEL_FAMILY のテスト
# ==============================


class TestTaskRouting:
    def test_default_routes_to_openai(self):
        assert TASK_TO_MODEL_FAMILY["default"] == "openai"

    def test_analysis_routes_to_gemini(self):
        assert TASK_TO_MODEL_FAMILY["analysis"] == "gemini"

    def test_code_routes_to_deepseek(self):
        assert TASK_TO_MODEL_FAMILY["code"] == "deepseek"

    def test_summarize_routes_to_openai(self):
        assert TASK_TO_MODEL_FAMILY["summarize"] == "openai"


# ==============================
# _raw_call のテスト
# ==============================


class TestRawCall:
    def _make_ctrl(self, **kwargs):
        ctrl = object.__new__(AILlmController)
        ctrl._initialized = True
        ctrl.gemini_model = kwargs.get("gemini_model")
        ctrl.deepseek_client = kwargs.get("deepseek_client")
        ctrl.openai_client = kwargs.get("openai_client")
        return ctrl

    def test_gemini_call(self):
        ctrl = self._make_ctrl(gemini_model=MagicMock())
        ctrl.gemini_model.generate_content.return_value = MagicMock(text="Gemini response")

        text, usage = ctrl._raw_call("gemini", "hello", None)

        assert text == "Gemini response"
        assert "input" in usage
        assert "output" in usage

    def test_deepseek_call(self):
        mock_client = MagicMock()
        mock_cmpl = MagicMock()
        mock_cmpl.choices = [MagicMock(message=MagicMock(content="DeepSeek response"))]
        mock_cmpl.usage.model_dump.return_value = {"prompt_tokens": 100, "completion_tokens": 50}
        mock_client.chat.completions.create.return_value = mock_cmpl

        ctrl = self._make_ctrl(deepseek_client=mock_client)
        text, usage = ctrl._raw_call("deepseek", "hello", [{"name": "tool1"}])

        assert text == "DeepSeek response"
        assert usage["prompt_tokens"] == 100

    def test_openai_call(self):
        mock_client = MagicMock()
        mock_cmpl = MagicMock()
        mock_cmpl.choices = [MagicMock(message=MagicMock(content="OpenAI response"))]
        mock_cmpl.usage.model_dump.return_value = {"prompt_tokens": 200, "completion_tokens": 80}
        mock_client.chat.completions.create.return_value = mock_cmpl

        ctrl = self._make_ctrl(openai_client=mock_client)
        text, usage = ctrl._raw_call("openai", "hello", None)

        assert text == "OpenAI response"
        assert usage["prompt_tokens"] == 200

    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA")
    def test_local_call(self, mock_llama):
        mock_llama.return_value = {"choices": [{"text": "Local response"}]}

        ctrl = self._make_ctrl()
        text, usage = ctrl._raw_call("local", "hello", None)

        assert text == "Local response"

    def test_no_client_raises(self):
        ctrl = self._make_ctrl()

        with pytest.raises(RuntimeError, match="No client configured"):
            ctrl._raw_call("gemini", "hello", None)

    def test_deepseek_empty_content_returns_empty_string(self):
        mock_client = MagicMock()
        mock_cmpl = MagicMock()
        mock_cmpl.choices = [MagicMock(message=MagicMock(content=None))]
        mock_cmpl.usage = None
        mock_client.chat.completions.create.return_value = mock_cmpl

        ctrl = self._make_ctrl(deepseek_client=mock_client)
        text, usage = ctrl._raw_call("deepseek", "hello", None)

        assert text == ""
        assert usage == {}


# ==============================
# _stream_call のテスト
# ==============================


class TestStreamCall:
    def _make_ctrl(self, **kwargs):
        ctrl = object.__new__(AILlmController)
        ctrl._initialized = True
        ctrl.openai_client = kwargs.get("openai_client")
        return ctrl

    def test_openai_streaming(self):
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hel"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="lo!"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        ctrl = self._make_ctrl(openai_client=mock_client)
        collected = []
        text, usage = ctrl._stream_call("openai", "test", lambda c: collected.append(c))

        assert text == "Hello!"
        assert collected == ["Hel", "lo!", ""]

    def test_non_openai_raises(self):
        ctrl = self._make_ctrl()

        with pytest.raises(RuntimeError, match="Streaming is only supported for OpenAI"):
            ctrl._stream_call("deepseek", "test", lambda c: None)

    def test_no_client_raises(self):
        ctrl = self._make_ctrl()

        with pytest.raises(RuntimeError):
            ctrl._stream_call("openai", "test", lambda c: None)


# ==============================
# _save_chat_history のテスト
# ==============================


class TestSaveChatHistory:
    def _make_ctrl(self):
        ctrl = object.__new__(AILlmController)
        ctrl._initialized = True
        return ctrl

    def test_import_error_skips_silently(self):
        ctrl = self._make_ctrl()
        result = GenerateResult(text="test", tokens={"input": 10, "output": 5})

        with patch.dict("sys.modules", {"app": None, "app.models": None}):
            ctrl._save_chat_history("hello", result)

    @patch("app.utils.ai_llm_controller.datetime")
    def test_saves_user_and_assistant_messages(self, mock_dt):
        mock_dt.utcnow.return_value = "2026-05-10T00:00:00"
        mock_db = MagicMock()
        mock_chat_history = MagicMock()

        ctrl = self._make_ctrl()
        result = GenerateResult(
            text="AI response",
            tokens={"input": 100, "output": 50},
            cost_usd=0.01,
            model_family="openai",
        )

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(db=mock_db),
                "app.models": MagicMock(ChatHistory=mock_chat_history),
            },
        ):
            ctrl._save_chat_history("hello", result)

        assert mock_db.session.add.call_count == 2
        mock_db.session.commit.assert_called_once()

    @patch("app.utils.ai_llm_controller.datetime")
    def test_db_error_rolls_back(self, mock_dt):
        mock_dt.utcnow.return_value = "2026-05-10T00:00:00"
        mock_db = MagicMock()
        mock_db.session.commit.side_effect = RuntimeError("DB error")

        ctrl = self._make_ctrl()
        result = GenerateResult(text="test", tokens={})

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(db=mock_db),
                "app.models": MagicMock(ChatHistory=MagicMock()),
            },
        ):
            ctrl._save_chat_history("hello", result)

        mock_db.session.rollback.assert_called_once()

    def test_handles_int_tokens(self):
        mock_db = MagicMock()

        # ChatHistory が実際の kwargs を保持するようにする
        created = []

        def fake_chat_history(**kwargs):
            m = MagicMock()
            m._kwargs = kwargs
            for k, v in kwargs.items():
                setattr(m, k, v)
            created.append(m)
            return m

        ctrl = self._make_ctrl()
        result = GenerateResult(text="test", tokens=150)

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(db=mock_db),
                "app.models": MagicMock(ChatHistory=fake_chat_history),
            },
        ):
            ctrl._save_chat_history("hello", result)

        # 2件目が assistant メッセージ
        assert created[1]._kwargs.get("tokens") == 150


# ==============================
# generate（統合）のテスト
# ==============================


class TestGenerate:
    def _make_ctrl(self):
        ctrl = object.__new__(AILlmController)
        ctrl._initialized = True
        return ctrl

    @patch("app.utils.ai_llm_controller.CACHE")
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA", None)
    def test_cache_hit_returns_cached(self, mock_cache):
        mock_cache.get.return_value = {
            "text": "cached result",
            "tokens": {"input": 10, "output": 5},
            "cost_usd": 0.001,
            "model_family": "openai",
            "sentiment": "NEUTRAL",
        }
        ctrl = self._make_ctrl()

        with patch.object(ctrl, "_save_chat_history"):
            result = ctrl.generate("hello")

        assert result.text == "cached result"
        assert result.cached is True

    @patch("app.utils.ai_llm_controller.CACHE")
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA", None)
    @patch.object(AILlmController, "_generate_with_retry")
    def test_cache_miss_generates_and_saves(self, mock_retry, mock_cache):
        mock_cache.get.return_value = None
        mock_retry.return_value = GenerateResult(
            text="new result",
            tokens={"input": 100, "output": 50},
            cost_usd=0.005,
            model_family="openai",
        )
        ctrl = self._make_ctrl()

        with patch.object(ctrl, "_save_chat_history") as mock_save:
            result = ctrl.generate("hello")

        assert result.text == "new result"
        assert result.cached is False
        mock_cache.set.assert_called_once()
        mock_save.assert_called_once()

    @patch("app.utils.ai_llm_controller.CACHE")
    @patch("app.utils.ai_llm_controller.LOCAL_NLP", None)
    @patch("app.utils.ai_llm_controller.LOCAL_LLAMA", None)
    @patch.object(AILlmController, "_generate_with_retry")
    def test_task_type_selects_family(self, mock_retry, mock_cache):
        mock_cache.get.return_value = None
        mock_retry.return_value = GenerateResult(text="ok", model_family="gemini")
        ctrl = self._make_ctrl()

        with patch.object(ctrl, "_save_chat_history"):
            ctrl.generate("analyze this", task_type="analysis")

        mock_retry.assert_called_once()
        assert mock_retry.call_args[0][0] == "gemini"


# ==============================
# Singleton のテスト
# ==============================


class TestSingleton:
    def test_new_returns_same_instance(self):
        AILlmController._instance = None

        with patch.object(AILlmController, "__init__", return_value=None):
            a = AILlmController()
            b = AILlmController()

        assert a is b

        AILlmController._instance = None
