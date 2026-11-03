"""Answer-only validation and test loss."""

import tempfile

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from src.config import Config


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    config: Config,
) -> float:
    """Compute the answer-only cross-entropy loss of a model with SFTTrainer.

    Uses the same evaluation batch size and maximum length as training, so
    the result is comparable with the losses logged during the searches.

    Args:
        model: Model to evaluate (its weights are not changed).
        tokenizer: Tokenizer matching the model.
        dataset: Prompt-completion dataset (see `src.formatting.to_sft_format`).
        config: Experiment configuration.

    Returns:
        The mean evaluation loss.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_eval_batch_size=config.training.eval_batch_size,
            completion_only_loss=True,
            max_length=config.model.max_seq_len,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            # SFTTrainer requires a train_dataset; nothing is trained here
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
        )
        metrics: dict[str, float] = trainer.evaluate()

    return metrics["eval_loss"]
