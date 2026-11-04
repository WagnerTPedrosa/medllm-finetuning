from pathlib import Path

import pytest

from medllm.config import (
    FloatRange,
    MultitaskConfig,
    PretrainConfig,
    compose_config,
    load_multitask_config,
    load_pretrain_config,
    parse_search_space,
)


def test_parse_search_space_reads_ranges_and_choices() -> None:
    raw_space = {
        "learning_rate": {"low": 1e-5, "high": 1e-4, "log": True},
        "batch_size": [2, 8],
    }

    search_space = parse_search_space(raw_space)

    assert search_space["learning_rate"] == FloatRange(low=1e-5, high=1e-4, log=True)
    assert search_space["batch_size"] == [2, 8]


def test_parse_search_space_rejects_non_numeric_range() -> None:
    with pytest.raises(ValueError):
        parse_search_space({"learning_rate": {"low": "1e-5", "high": 1e-4}})


MODEL_FILES: list[Path] = sorted(Path("configs/model").glob("*.yaml"))


def test_compose_config_defaults_to_qwen() -> None:
    raw = compose_config()

    assert raw["model"]["name"] == "qwen3-1.7b"
    assert raw["results_dir"] == "results/qwen3-1.7b"


@pytest.mark.parametrize("model_file", MODEL_FILES, ids=[path.stem for path in MODEL_FILES])
def test_every_model_file_fills_the_config_and_is_named_after_its_file(model_file: Path) -> None:
    # compose_config raises if a `???` value is left unset
    raw = compose_config([f"model={model_file.stem}"])

    assert raw["model"]["name"] == model_file.stem


def test_compose_config_rejects_an_unknown_model() -> None:
    with pytest.raises(Exception):
        compose_config(["model=not-a-model"])


def test_load_pretrain_config_names_mlflow_and_paths_after_the_model() -> None:
    config: PretrainConfig = load_pretrain_config(["model=qwen3-1.7b"])

    assert config.model.model_id == "Qwen/Qwen3-1.7B-Base"
    assert config.mlflow.experiment_name == "qwen3-1.7b-pubmed-pretrain"
    assert config.mlflow.registered_model_name == "qwen3-1.7b-pubmed-dapt"
    assert config.corpus.processed_dir == "data/processed/pubmed_qwen3-1.7b"
    assert config.pretraining.output_dir == "checkpoints/pretrain/qwen3-1.7b-pubmed-dapt"
    assert config.results_dir == "results/qwen3-1.7b"


def test_load_multitask_config_reads_the_lora_settings() -> None:
    finetune = load_multitask_config().finetune

    assert finetune.method == "lora"
    assert finetune.seeds == [42]
    assert finetune.lora.r == 16
    assert "q_proj" in finetune.lora.target_modules


def test_load_multitask_config_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="finetune.method"):
        load_multitask_config(["finetune.method=qlora"])


def test_load_pretrain_config_expands_the_corpus_directory() -> None:
    config: PretrainConfig = load_pretrain_config()

    assert not config.corpus.raw_dir.startswith("~")


def test_load_multitask_config_points_dapt_at_the_pretrained_model() -> None:
    config: MultitaskConfig = load_multitask_config(["model=qwen3-1.7b"])

    assert config.mlflow.experiment_name == "qwen3-1.7b-med-multitask"
    assert config.mlflow.registered_model_name == "qwen3-1.7b-med-multitask"
    assert config.finetune.dapt_model_id == "models:/qwen3-1.7b-pubmed-dapt@candidate"


def test_fine_tuning_data_is_filtered_with_the_model_tokenizer() -> None:
    qwen: MultitaskConfig = load_multitask_config(["model=qwen3-1.7b"])

    assert qwen.data_filter.tokenizer == qwen.model.model_id
    assert qwen.data_filter.max_seq_len == qwen.model.max_seq_len
    assert qwen.finetune_data.processed_dir == "data/processed/finetune_qwen3-1.7b"
    assert qwen.training.max_epochs == qwen.mixture.epochs


def test_gpt2_keeps_the_archived_v1_data() -> None:
    gpt2: MultitaskConfig = load_multitask_config(["model=gpt2"])

    assert gpt2.finetune_data.processed_dir == "data/processed/multitask_v1"
    assert load_pretrain_config(["model=gpt2"]).corpus.processed_dir == "data/processed/pubmed_v1_gpt2"
    assert gpt2.data_filter.tokenizer == "openai-community/gpt2"
    assert gpt2.finetune.method == "full"


def test_load_multitask_config_merges_the_model_search_range_into_the_sweep() -> None:
    sweep = load_multitask_config(["model=qwen3-1.7b"]).sweep

    assert sweep.search_space["learning_rate"] == FloatRange(low=5e-6, high=1e-4, log=True)
    assert sweep.search_space["batch_size"] == [8, 16, 32]


def test_every_raw_dataset_is_pinned_to_a_revision() -> None:
    config: MultitaskConfig = load_multitask_config()

    assert set(config.datasets.sources) == {"medtext", "pubmedqa_labeled", "pubmedqa_artificial", "medmcqa", "medqa"}
    assert config.datasets.sources["pubmedqa_artificial"].subset == "pqa_artificial"
    assert all(len(source.revision) == 40 for source in config.datasets.sources.values())
