"""LLM adapter registry for Part 2 experimentation."""

from __future__ import annotations

import logging
import os
import random
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

    def _prepare_inputs(self, prompt: str):
        tokenizer_kwargs = {"return_tensors": "pt"}
        if self._max_input_tokens:
            token_count = len(self._tokenizer.encode(prompt, add_special_tokens=False))
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
    raise KeyError(f"Unknown adapter: {name}")


__all__ = [
    "TextAdapter",
    "FlanT5Adapter",
    "HuggingFaceCausalAdapter",
    "OpenAIChatAdapter",
    "create_adapter",
]

