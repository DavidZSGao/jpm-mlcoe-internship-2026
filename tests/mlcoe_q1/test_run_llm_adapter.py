import json

import pandas as pd

from mlcoe_q1.pipelines import run_llm_adapter
from mlcoe_q1.llm.adapters import FlanT5Adapter


class DummyAdapter:
    name = "dummy"
    model_id = "dummy-model"

    def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        assert "Respond with JSON only" in prompt
        return json.dumps(
            {
                "balance_sheet": {"cash": 1000},
                "income_statement": {"net_income": 200},
            }
        )


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
    payload = json.loads(output.loc[0, "response"])
    assert payload["balance_sheet"]["cash"] == 1000
    assert payload["income_statement"]["net_income"] == 200


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

