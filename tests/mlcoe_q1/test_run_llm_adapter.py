import json
import sys
import types

import pandas as pd

from mlcoe_q1.pipelines import run_llm_adapter
from mlcoe_q1.llm import adapters
from mlcoe_q1.llm.adapters import (
    FlanT5Adapter,
    HuggingFaceCausalAdapter,
    OpenAIChatAdapter,
    create_adapter,
)


class DummyAdapter:
    name = "dummy"
    model_id = "dummy-model"

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        assert "Respond with JSON only" in prompt
        return json.dumps(
            {
                "balance_sheet": {"cash": 1000},
                "income_statement": {"net_income": 200},
            }
        )

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)


def test_run_llm_adapter_writes_responses(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "context_period": ["2023-12-31"],
            "target_period": ["2024-12-31"],
            "statements": [["balance_sheet", "income_statement"]],
            "prompt": ["Return forecasts."],
        }
    )
    prompts_path = tmp_path / "prompts.parquet"
    data.to_parquet(prompts_path)

    monkeypatch.setattr(run_llm_adapter, "create_adapter", lambda *args, **kwargs: DummyAdapter())

    output_path = tmp_path / "responses.parquet"
    run_llm_adapter.main(
        [
            "--prompts",
            str(prompts_path),
            "--output",
            str(output_path),
            "--adapter",
            "dummy",
        ]
    )

    output = pd.read_parquet(output_path)
    assert output.loc[0, "adapter"] == "dummy"
    assert pd.isna(output.loc[0, "seed"])
    payload = json.loads(output.loc[0, "response"])
    assert payload["balance_sheet"]["cash"] == 1000
    assert payload["income_statement"]["net_income"] == 200

    metadata_path = output_path.with_name(output_path.name + ".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["adapter"] == "dummy"
    assert metadata["models"] == ["dummy-model"]
    assert metadata["seeds"] == [None]
    assert metadata["records"] == 1
    assert metadata["api_base"] is None
    assert metadata["api_key_env"] is None


def test_run_llm_adapter_supports_config(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "context_period": ["2023-12-31"],
            "target_period": ["2024-12-31"],
            "statements": [["balance_sheet"]],
            "prompt": ["Return forecasts."],
        }
    )
    prompts_path = tmp_path / "prompts.parquet"
    data.to_parquet(prompts_path)

    monkeypatch.setattr(run_llm_adapter, "create_adapter", lambda *args, **kwargs: DummyAdapter())

    output_path = tmp_path / "responses.parquet"
    metadata_path = tmp_path / "responses.meta.json"

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "prompts": str(prompts_path),
                "output": str(output_path),
                "adapter": "dummy",
                "metadata_output": str(metadata_path),
            }
        ),
        encoding="utf-8",
    )

    run_llm_adapter.main(["--config", str(config_path)])

    assert output_path.exists()
    assert metadata_path.exists()


def test_run_llm_adapter_handles_models_and_seeds(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "context_period": ["2023-12-31"],
            "target_period": ["2024-12-31"],
            "statements": [["balance_sheet"]],
            "prompt": ["Return forecasts."],
        }
    )
    prompts_path = tmp_path / "prompts.parquet"
    data.to_parquet(prompts_path)

    created: list[DummyAdapter] = []

    def factory(*args, **kwargs):
        adapter = DummyAdapter()
        adapter.model_id = kwargs.get("model_name") or "default"
        created.append(adapter)
        return adapter

    monkeypatch.setattr(run_llm_adapter, "create_adapter", factory)

    output_path = tmp_path / "responses.parquet"
    run_llm_adapter.main(
        [
            "--prompts",
            str(prompts_path),
            "--output",
            str(output_path),
            "--adapter",
            "dummy",
            "--model",
            "model-a,model-b",
            "--seeds",
            "1,2",
        ]
    )

    output = pd.read_parquet(output_path)
    assert set(output["model"]) == {"model-a", "model-b"}
    assert set(output["seed"]) == {1, 2}
    assert len(output) == 4  # 2 models * 2 seeds
    assert all(adapter.seeds == [1, 2] for adapter in created)

    metadata_path = output_path.with_name(output_path.name + ".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["models"] == ["model-a", "model-b"]
    assert metadata["seeds"] == [1, 2]
    assert metadata["records"] == 4
    assert metadata["model_argument"] == "model-a,model-b"
    assert metadata["seed_argument"] == "1,2"


def test_flan_t5_adapter_truncates_long_prompts(monkeypatch):
    adapter = FlanT5Adapter.__new__(FlanT5Adapter)
    adapter.model_name = "stub"
    adapter.max_new_tokens = 16
    adapter.name = "flan-t5"
    adapter.model_id = "stub"

    class DummyTokenizer:
        def __init__(self):
            self.model_max_length = 5
            self.calls = []

        def encode(self, prompt: str, add_special_tokens: bool = True):
            return list(range(len(prompt.split())))

        def __call__(self, prompt: str, **kwargs):
            self.calls.append(kwargs)
            return {"input_ids": [[0, 1, 2]], "attention_mask": [[1, 1, 1]]}

        def decode(self, ids, skip_special_tokens: bool = True):
            return "decoded"

    class DummyOutputs:
        def __getitem__(self, index):
            return [0, 1, 2]

    class DummyModel:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return DummyOutputs()

    tokenizer = DummyTokenizer()
    adapter._tokenizer = tokenizer
    adapter._model = DummyModel()
    adapter._max_input_tokens = tokenizer.model_max_length

    prompt = " ".join(f"token{i}" for i in range(20))
    adapter.generate(prompt)

    kwargs = tokenizer.calls[-1]
    assert kwargs.get("truncation") is True
    assert kwargs.get("max_length") == tokenizer.model_max_length


def test_hf_causal_adapter_sets_pad_and_sampling():
    adapter = HuggingFaceCausalAdapter.__new__(HuggingFaceCausalAdapter)
    adapter.model_name = "stub"
    adapter.max_new_tokens = 8
    adapter.name = "hf-causal"
    adapter.model_id = "stub"

    class DummyTokenizer:
        def __init__(self):
            self.model_max_length = 4
            self.pad_token_id = None
            self.eos_token_id = 7
            self.calls = []

        def encode(self, prompt: str, add_special_tokens: bool = False):
            return list(range(len(prompt.split())))

        def __call__(self, prompt: str, **kwargs):
            self.calls.append(kwargs)
            return {"input_ids": [[0, 1, 2]], "attention_mask": [[1, 1, 1]]}

        def decode(self, ids, skip_special_tokens: bool = True):
            return "decoded"

    class DummyModel:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return [[0, 1, 2, 3]]

    tokenizer = DummyTokenizer()
    adapter._tokenizer = tokenizer
    adapter._model = DummyModel()
    adapter._max_input_tokens = tokenizer.model_max_length

    prompt = " ".join(f"token{i}" for i in range(20))
    adapter.generate(prompt, temperature=0.7)

    kwargs = tokenizer.calls[-1]
    assert kwargs.get("truncation") is True
    assert kwargs.get("max_length") == tokenizer.model_max_length
    assert adapter._tokenizer.pad_token_id == adapter._tokenizer.eos_token_id

    model_kwargs = adapter._model.kwargs
    assert model_kwargs["pad_token_id"] == adapter._tokenizer.eos_token_id
    assert model_kwargs["do_sample"] is True


def test_create_adapter_returns_hf_causal(monkeypatch):
    class DummyAdapter:
        def __init__(self, *, model_name: str, max_new_tokens: int):
            self.model_name = model_name
            self.max_new_tokens = max_new_tokens

    monkeypatch.setattr(adapters, "HuggingFaceCausalAdapter", DummyAdapter)

    adapter = create_adapter("hf-causal", model_name="distilgpt2", max_new_tokens=64)
    assert isinstance(adapter, DummyAdapter)
    assert adapter.model_name == "distilgpt2"
    assert adapter.max_new_tokens == 64


def test_create_openai_chat_adapter_uses_environment(monkeypatch):
    class DummyCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = types.SimpleNamespace(content=" result ")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = DummyChat()

    dummy_module = types.SimpleNamespace(OpenAI=DummyClient)
    monkeypatch.setitem(sys.modules, 'openai', dummy_module)
    monkeypatch.setenv('CUSTOM_OPENAI', 'secret-key')

    adapter = OpenAIChatAdapter(
        model_name='stub-model',
        api_key_env='CUSTOM_OPENAI',
        api_base='https://example.test',
        request_timeout=5.0,
    )
    output = adapter.generate('Prompt text', temperature=0.3, max_new_tokens=42)

    assert output == 'result'
    assert adapter.name == 'openai-chat'
    assert adapter.model_id == 'stub-model'
    assert adapter._client.kwargs['api_key'] == 'secret-key'
    assert adapter._client.kwargs['base_url'] == 'https://example.test'
    assert adapter._client.chat.completions.kwargs['timeout'] == 5.0
    assert adapter._client.chat.completions.kwargs['max_tokens'] == 42

