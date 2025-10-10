"""LLM adapter registry for Part 2 experimentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol


class TextAdapter(Protocol):
    """Protocol for simple text-generation adapters."""

    name: str
    model_id: str

    def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        """Return the generated completion for ``prompt``."""


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


def create_adapter(name: str, **kwargs) -> TextAdapter:
    """Instantiate an adapter by registry name."""

    key = name.lower()
    if key in {"flan", "flan-t5"}:
        model_name = kwargs.get("model_name") or kwargs.get("model") or "google/flan-t5-small"
        max_new_tokens = kwargs.get("max_new_tokens", 256)
        return FlanT5Adapter(model_name=model_name, max_new_tokens=max_new_tokens)
    raise KeyError(f"Unknown adapter: {name}")


__all__ = ["TextAdapter", "FlanT5Adapter", "create_adapter"]

