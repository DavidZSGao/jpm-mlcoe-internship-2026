"""LLM adapter registry for Part 2 experimentation."""

from __future__ import annotations

import json
import http.client
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TextAdapter(Protocol):
    """Protocol for simple text-generation adapters."""

    name: str
    model_id: str

    def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        """Return the generated completion for ``prompt``."""

    def set_seed(self, seed: int) -> None:
        """Configure adapter-level randomness (optional)."""


@dataclass
class FlanT5Adapter:
    """TensorFlow-backed adapter for FLAN-T5 models via ``transformers``."""

    model_name: str = "google/flan-t5-small"
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy import

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to('cpu')
        self.name = "flan-t5"
        self.model_id = self.model_name
        max_length = getattr(self._tokenizer, "model_max_length", None)
        if max_length is not None and (max_length <= 0 or max_length > 1_000_000):
            max_length = None
        self._max_input_tokens = max_length

    def _prepare_inputs(self, prompt: str):
        tokenizer_kwargs = {"return_tensors": "pt"}
        if self._max_input_tokens:
            token_count = len(self._tokenizer.encode(prompt, add_special_tokens=True))
            if token_count > self._max_input_tokens:
                logging.warning(
                    "Prompt length %d exceeds model limit %d for %s; truncating",
                    token_count,
                    self._max_input_tokens,
                    self.model_id,
                )
                tokenizer_kwargs.update(
                    {"truncation": True, "max_length": self._max_input_tokens}
                )
        return self._tokenizer(prompt, **tokenizer_kwargs)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> str:
        max_tokens = max_new_tokens or self.max_new_tokens
        do_sample = temperature > 0
        inputs = self._prepare_inputs(prompt)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=float(temperature),
            do_sample=do_sample,
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ModuleNotFoundError:
            pass
        self._seed = seed


@dataclass
class HuggingFaceCausalAdapter:
    """Adapter for causal language models via ``transformers``."""

    model_name: str = "distilgpt2"
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            eos = getattr(self._tokenizer, "eos_token_id", None)
            if eos is not None:
                self._tokenizer.pad_token_id = eos
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.to('cpu')
        self.name = "hf-causal"
        self.model_id = self.model_name
        max_length = getattr(self._tokenizer, "model_max_length", None)
        if max_length is not None and (max_length <= 0 or max_length > 1_000_000):
            max_length = None
        self._max_input_tokens = max_length

    def _prepare_inputs(self, prompt: str, max_input_tokens: int | None = None):
        tokenizer_kwargs = {"return_tensors": "pt"}
        max_length = max_input_tokens or self._max_input_tokens
        if max_length:
            token_count = len(self._tokenizer.encode(prompt, add_special_tokens=False))
            if token_count > max_length:
                logging.warning(
                    "Prompt length %d exceeds model limit %d for %s; truncating",
                    token_count,
                    max_length,
                    self.model_id,
                )
                tokenizer_kwargs.update({"truncation": True, "max_length": max_length})
        return self._tokenizer(prompt, **tokenizer_kwargs)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> str:
        max_tokens = max_new_tokens or self.max_new_tokens
        do_sample = temperature > 0
        max_input_tokens = None
        if self._max_input_tokens:
            max_input_tokens = max(self._max_input_tokens - max_tokens, 1)
        inputs = self._prepare_inputs(prompt, max_input_tokens=max_input_tokens)
        max_length = getattr(self._model.config, "max_position_embeddings", None)
        if max_length is not None:
            input_len = int(inputs["input_ids"].shape[-1])
            available = max_length - input_len
            if available <= 0:
                logging.warning(
                    "Input length %d exceeds model max %d for %s; forcing max_new_tokens=1",
                    input_len,
                    max_length,
                    self.model_id,
                )
                max_tokens = 1
            elif max_tokens > available:
                logging.warning(
                    "Reducing max_new_tokens from %d to %d to fit model limit %d for %s",
                    max_tokens,
                    available,
                    max_length,
                    self.model_id,
                )
                max_tokens = available
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            eos = getattr(self._tokenizer, "eos_token_id", None)
            pad_token_id = eos
            if eos is not None:
                try:
                    self._tokenizer.pad_token_id = eos
                except AttributeError:
                    pass
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=float(temperature),
            do_sample=do_sample,
            pad_token_id=pad_token_id,
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ModuleNotFoundError:
            pass
        self._seed = seed


@dataclass
class OpenAIChatAdapter:
    """Adapter for hosted OpenAI chat-completion models."""

    model_name: str = "gpt-4o-mini"
    max_new_tokens: int = 512
    api_key_env: str = "OPENAI_API_KEY"
    api_base: str | None = None
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore import-not-found
        except ModuleNotFoundError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError(
                "The openai package is required for OpenAIChatAdapter"
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} must be set for OpenAIChatAdapter"
            )

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        self._client = OpenAI(**client_kwargs)
        self.name = "openai-chat"
        self.model_id = self.model_name

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> str:
        max_tokens = max_new_tokens or self.max_new_tokens
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=self.request_timeout,
            )
        except Exception as exc:
            logging.warning(
                "OpenAI chat request with response_format failed (%s); retrying without it",
                exc,
            )
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
                max_tokens=max_tokens,
                timeout=self.request_timeout,
            )
        choices = getattr(response, "choices", [])
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return str(content or "").strip()

    def set_seed(self, seed: int) -> None:
        # The OpenAI Chat API is not seedable, but we record the value for transparency.
        self._seed = seed


@dataclass
class OpenAIResponsesAdapter:
    """Adapter for hosted OpenAI models via the Responses API."""

    model_name: str = "gpt-4o-mini"
    max_new_tokens: int = 512
    api_key_env: str = "OPENAI_API_KEY"
    api_base: str | None = None
    request_timeout: float = 60.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore import-not-found
        except ModuleNotFoundError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError(
                "The openai package is required for OpenAIResponsesAdapter"
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} must be set for OpenAIResponsesAdapter"
            )
        self._api_key = api_key

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        self._client = OpenAI(**client_kwargs)
        self.name = "openai-responses"
        self.model_id = self.model_name

    def _extract_text(self, response: object) -> str:
        if isinstance(response, dict):
            output_text = response.get("output_text")
            if isinstance(output_text, str):
                return output_text.strip()
            output = response.get("output")
            if isinstance(output, list):
                for item in output:
                    content = item.get("content") if isinstance(item, dict) else None
                    if isinstance(content, list):
                        for chunk in content:
                            if not isinstance(chunk, dict):
                                continue
                            if chunk.get("type") in {"output_text", "text"}:
                                text = chunk.get("text")
                                if text:
                                    return str(text).strip()
            return ""
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text.strip()
        output = getattr(response, "output", None)
        if isinstance(output, list):
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for chunk in content:
                        if getattr(chunk, "type", None) in {"output_text", "text"}:
                            text = getattr(chunk, "text", "")
                            if text:
                                return str(text).strip()
        return ""

    def _responses_url(self) -> str:
        if self.api_base:
            base = self.api_base.rstrip("/")
            if base.endswith("/v1"):
                return f"{base}/responses"
            return f"{base}/v1/responses"
        return "https://api.openai.com/v1/responses"

    def _request_responses_http(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = {
            "model": self.model_name,
            "input": prompt,
            "temperature": float(temperature),
            "max_output_tokens": int(max_output_tokens),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._responses_url(),
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        attempt = 0
        last_error: Exception | None = None
        while attempt < max(1, self.max_retries):
            attempt += 1
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    response = json.loads(resp.read().decode("utf-8", errors="ignore"))
                return self._extract_text(response)
            except urllib.error.HTTPError as exc:
                status = getattr(exc, "code", None)
                if status in {400, 401, 403}:
                    raise
                last_error = exc
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionResetError,
                http.client.RemoteDisconnected,
            ) as exc:
                last_error = exc
            if attempt < max(1, self.max_retries):
                sleep_s = min(2 ** (attempt - 1), 8) + random.random()
                time.sleep(sleep_s)
        raise RuntimeError(
            f"OpenAI Responses HTTP request failed after {attempt} attempts"
        ) from last_error

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> str:
        max_tokens = max_new_tokens or self.max_new_tokens
        responses_client = getattr(self._client, "responses", None)
        if responses_client is not None:
            response = responses_client.create(
                model=self.model_name,
                input=prompt,
                temperature=float(temperature),
                max_output_tokens=max_tokens,
                timeout=self.request_timeout,
            )
            return self._extract_text(response)
        return self._request_responses_http(
            prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def set_seed(self, seed: int) -> None:
        # The OpenAI Responses API is not seedable, but we record the value for transparency.
        self._seed = seed


def create_adapter(name: str, **kwargs) -> TextAdapter:
    """Instantiate an adapter by registry name."""

    key = name.lower()
    if key in {"flan", "flan-t5"}:
        model_name = kwargs.get("model_name") or kwargs.get("model") or "google/flan-t5-small"
        max_new_tokens = kwargs.get("max_new_tokens", 256)
        return FlanT5Adapter(model_name=model_name, max_new_tokens=max_new_tokens)
    if key in {"hf-causal", "causal"}:
        model_name = kwargs.get("model_name") or kwargs.get("model") or "distilgpt2"
        max_new_tokens = kwargs.get("max_new_tokens", 256)
        return HuggingFaceCausalAdapter(model_name=model_name, max_new_tokens=max_new_tokens)
    if key in {"openai", "openai-chat"}:
        model_name = kwargs.get("model_name") or kwargs.get("model") or "gpt-4o-mini"
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        api_key_env = kwargs.get("api_key_env") or kwargs.get("api_key_variable") or "OPENAI_API_KEY"
        api_base = kwargs.get("api_base") or kwargs.get("base_url")
        request_timeout = kwargs.get("request_timeout", 60.0)
        return OpenAIChatAdapter(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            api_key_env=api_key_env,
            api_base=api_base,
            request_timeout=float(request_timeout),
        )
    if key in {"openai-responses", "openai-response", "responses"}:
        model_name = kwargs.get("model_name") or kwargs.get("model") or "gpt-4o-mini"
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        api_key_env = kwargs.get("api_key_env") or kwargs.get("api_key_variable") or "OPENAI_API_KEY"
        api_base = kwargs.get("api_base") or kwargs.get("base_url")
        request_timeout = kwargs.get("request_timeout", 60.0)
        return OpenAIResponsesAdapter(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            api_key_env=api_key_env,
            api_base=api_base,
            request_timeout=float(request_timeout),
        )
    raise KeyError(f"Unknown adapter: {name}")


__all__ = [
    "TextAdapter",
    "FlanT5Adapter",
    "HuggingFaceCausalAdapter",
    "OpenAIChatAdapter",
    "OpenAIResponsesAdapter",
    "create_adapter",
]
