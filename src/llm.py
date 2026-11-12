"""LLM factory. The model, provider and limits come from Settings, never from code.

Providers:
  bedrock   — Amazon Bedrock via IAM (AWS credentials / runtime role). No Anthropic key.
  anthropic — Claude API. Auth, in order: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, then the
              `ant auth login` profile (local development only; tokens are short-lived and
              refreshed by the SDK). Billed to the Console organization, not a Claude Pro plan.
  xai       — xAI Grok API (OpenAI-compatible) via langchain-xai. Auth: XAI_API_KEY.
              LLM_EFFORT is not sent: not every Grok model accepts reasoning_effort.
  deepseek  — DeepSeek API (OpenAI format, https://api.deepseek.com) via langchain-deepseek.
              Auth: DEEPSEEK_API_KEY. Models: deepseek-v4-pro, deepseek-flash. Thinking is on by
              default; LLM_EFFORT=low|high|max sets reasoning_effort, LLM_EFFORT=none turns it off.
"""
from __future__ import annotations

import os
from functools import cached_property, lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.config import get_settings


def _anthropic_with_sdk_auth():
    """ChatAnthropic that lets the Anthropic SDK resolve credentials when no API key is set.

    langchain-anthropic always passes `api_key=` to the SDK client, which disables the SDK's own
    lookup (auth token, `ant auth login` profile). Leaving it out restores that lookup."""
    import anthropic
    from langchain_anthropic import ChatAnthropic
    from langchain_anthropic.chat_models import _get_default_async_httpx_client, _get_default_httpx_client

    class SdkAuthChatAnthropic(ChatAnthropic):
        @cached_property
        def _client_params(self) -> dict[str, Any]:
            key = self.anthropic_api_key.get_secret_value() if self.anthropic_api_key else ""
            if key:
                return super()._client_params
            headers = {"User-Agent": "langchain-anthropic"}
            if self.default_headers:
                headers.update(self.default_headers)
            params: dict[str, Any] = {"base_url": self.anthropic_api_url, "max_retries": self.max_retries,
                                      "default_headers": headers}
            if self.default_request_timeout is None or self.default_request_timeout > 0:
                params["timeout"] = self.default_request_timeout
            return params

        def _http_params(self) -> dict[str, Any]:
            p = {"base_url": self._client_params["base_url"]}
            if "timeout" in self._client_params:
                p["timeout"] = self._client_params["timeout"]
            if self.anthropic_proxy:
                p["anthropic_proxy"] = self.anthropic_proxy
            return p

        @cached_property
        def _client(self) -> anthropic.Client:
            return anthropic.Client(**self._client_params, http_client=_get_default_httpx_client(**self._http_params()))

        @cached_property
        def _async_client(self) -> anthropic.AsyncClient:
            return anthropic.AsyncClient(**self._client_params,
                                         http_client=_get_default_async_httpx_client(**self._http_params()))

    return SdkAuthChatAnthropic


def _deepseek_with_reasoning_replay():
    """ChatDeepSeek that sends each assistant turn's reasoning_content back to the API.

    In thinking mode with tools bound, DeepSeek requires every earlier assistant message to carry
    its reasoning_content, or it returns 400. langchain-deepseek stores it in
    AIMessage.additional_kwargs["reasoning_content"] but drops it from the request payload."""
    from langchain_core.messages import AIMessage
    from langchain_deepseek import ChatDeepSeek

    class ReasoningReplayChatDeepSeek(ChatDeepSeek):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            messages = self._convert_input(input_).to_messages()
            wire = payload.get("messages", [])
            if len(messages) != len(wire):  # payload no longer 1:1 with messages; leave it alone
                return payload
            for msg, out in zip(messages, wire):
                if isinstance(msg, AIMessage) and out.get("role") == "assistant":
                    reasoning = msg.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        out["reasoning_content"] = reasoning
            return payload

    return ReasoningReplayChatDeepSeek


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    s = get_settings()
    if s.llm_provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        kwargs = {}
        if s.llm_effort:
            # Turns on adaptive thinking with this effort (Opus 4.8 has thinking off by default;
            # on Opus 5.5 thinking is always on and effort defaults to medium).
            kwargs["reasoning_effort"] = s.llm_effort
        return ChatBedrockConverse(
            model=s.llm_model_id,
            region_name=s.aws_region,
            max_tokens=s.llm_max_tokens,
            **kwargs,
        )
    if s.llm_provider == "anthropic":
        kwargs = {}
        if s.llm_effort:
            kwargs["effort"] = s.llm_effort
            kwargs["thinking"] = {"type": "adaptive"}
        cls = _anthropic_with_sdk_auth()
        return cls(
            model=s.llm_model_id,
            max_tokens=s.llm_max_tokens,
            timeout=s.llm_timeout_s,
            max_retries=s.llm_max_retries,
            **kwargs,
        )
    if s.llm_provider == "xai":
        from langchain_xai import ChatXAI

        if not os.environ.get("XAI_API_KEY"):
            raise EnvironmentError("LLM_PROVIDER=xai needs XAI_API_KEY (in .env, or XAI_API_KEY_SECRET_ARN on AWS)")
        return ChatXAI(
            model=s.llm_model_id,
            max_tokens=s.llm_max_tokens,
            timeout=s.llm_timeout_s,
            max_retries=s.llm_max_retries,
        )
    if s.llm_provider == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise EnvironmentError(
                "LLM_PROVIDER=deepseek needs DEEPSEEK_API_KEY (in .env, or DEEPSEEK_API_KEY_SECRET_ARN on AWS)")
        kwargs = {}
        if s.llm_effort == "none":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        elif s.llm_effort:
            kwargs["reasoning_effort"] = s.llm_effort  # DeepSeek maps medium/xhigh to high
        cls = _deepseek_with_reasoning_replay()
        return cls(
            model=s.llm_model_id,
            max_tokens=s.llm_max_tokens,
            timeout=s.llm_timeout_s,
            max_retries=s.llm_max_retries,
            # Stream so bytes keep flowing while the model thinks over a long context; a silent
            # non-streaming request of 100k+ tokens can sit for minutes and get its connection dropped.
            streaming=True,
            **kwargs,
        )
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r}; use 'anthropic', 'bedrock', 'xai' or 'deepseek'")
