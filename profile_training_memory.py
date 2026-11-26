import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import batched
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from memory import GibMemoryUsage, measure_memory
from project_config import MODEL_ID, MODEL_REVISION
from report import write_memory_profile_report


@dataclass
class E2EMeasure:
    initial_memory: GibMemoryUsage | None = None
    base_model_initialized_memory: GibMemoryUsage | None = None
    base_model_trainable_params: int | None = None
    trainer_initialized_memory: GibMemoryUsage | None = None
    trainer_initialized_trainable_params: int | None = None
    after_forward_memory: GibMemoryUsage | None = None
    after_backward_memory: GibMemoryUsage | None = None
    after_optimizer_step_memory: GibMemoryUsage | None = None
    first_forward_token_count: int | None = None


class MemorySFTTrainer(SFTTrainer):
    class OptimizerStepCallback(TrainerCallback):
        def __init__(self, measure: E2EMeasure):
            self.measure = measure

        def on_optimizer_step(self, args, state, control, **kwargs):
            if self.measure.after_optimizer_step_memory is None:
                self.measure.after_optimizer_step_memory = measure_memory()

    def __init__(self, *args, measure: E2EMeasure, **kwargs):
        super().__init__(*args, **kwargs)
        self.measure = measure
        self._forward_hook_handle = self.model.register_forward_hook(
            self._measure_forward, with_kwargs=True
        )
        self.add_callback(self.OptimizerStepCallback(measure))

    def _measure_forward(self, module, args, kwargs, output):
        if module.training and self.measure.after_forward_memory is None:
            self.measure.after_forward_memory = measure_memory()
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None:
                self.measure.first_forward_token_count = int(attention_mask.sum().item())
            elif kwargs.get("input_ids") is not None:
                self.measure.first_forward_token_count = kwargs["input_ids"].numel()
            self._forward_hook_handle.remove()

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        if self.measure.after_backward_memory is None:
            self.measure.after_backward_memory = measure_memory()

        return loss


def profile_training_memory(
    *,
    dataset_path: str | Path,
    source_indices: list[int],
    model_id: str,
    revision: str,
    dtype: torch.dtype,
    sft_config_kwargs: dict,
    lora_config_kwargs: dict,
    output_path: str | Path,
    generate_html: bool = True,
) -> E2EMeasure:
    """Profile one CUDA optimizer step using zero-based JSONL row indices."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    dataset_path = Path(dataset_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path == dataset_path:
        raise ValueError("The profile output must differ from the source dataset.")
    if generate_html and output_path.with_suffix(".html").resolve() in (dataset_path, output_path):
        raise ValueError("The HTML report must have a separate output path.")
    if not source_indices:
        raise ValueError("At least one source index is required.")

    sft_config = SFTConfig(**{
        **sft_config_kwargs,
        "max_steps": 1,
        "warmup_steps": 0,
        "eval_strategy": "no",
        "eval_on_start": False,
        "save_strategy": "no",
    })
    lora_config = LoraConfig(**lora_config_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True,
    )

    dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
    )

    if any(index < 0 or index >= len(dataset) for index in source_indices):
        raise ValueError("Source indices must refer to rows in the source dataset.")
    train_dataset = dataset.select(source_indices)

    torch.cuda.reset_peak_memory_stats(device=0)
    measure = E2EMeasure(initial_memory=measure_memory())
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        dtype=dtype,
        device_map=0
    )
    measure.base_model_initialized_memory = measure_memory()
    measure.base_model_trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    trainer = MemorySFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        peft_config=lora_config,
        train_dataset=train_dataset,
        measure=measure,
    )
    measure.trainer_initialized_memory = measure_memory()
    measure.trainer_initialized_trainable_params = sum(
        p.numel()
        for p in trainer.model.parameters()
        if p.requires_grad
    )

    torch.cuda.reset_peak_memory_stats(device=0)
    trainer.train()

    resolved_lora_config = lora_config.to_dict()
    for key, value in resolved_lora_config.items():
        if isinstance(value, set):
            resolved_lora_config[key] = sorted(value)
    profile = {
        "dataset_path": str(dataset_path),
        "source_indices": source_indices,
        "model_id": model_id,
        "revision": revision,
        "dtype": str(dtype),
        "gpu_name": torch.cuda.get_device_name(0),
        "sft_config": sft_config.to_dict(),
        "lora_config": resolved_lora_config,
        "measurements": asdict(measure),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    if generate_html:
        write_memory_profile_report(output_path)
    return measure


def _find_longest_source_indices(dataset_path: Path, tokenizer, count: int) -> list[int]:
    lengths = []
    with dataset_path.open(encoding="utf-8") as source:
        rows = (json.loads(line) for line in source if line.strip())
        for batch in batched(enumerate(rows), 128):
            conversations = [row["prompt"] + row["completion"] for _, row in batch]
            token_ids = tokenizer.apply_chat_template(
                conversations, tokenize=True, return_dict=True, truncation=False,
            )["input_ids"]
            for (index, _), ids in zip(batch, token_ids, strict=True):
                lengths.append((index, len(ids)))

    if count > len(lengths):
        raise ValueError(f"Batch size {count} exceeds the dataset size ({len(lengths)}).")
    lengths.sort(key=lambda row: row[1], reverse=True)
    return [index for index, _ in lengths[:count]]


def main():
    parser = argparse.ArgumentParser(description="Profile one LoRA training step on the longest examples.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of examples per batch (default: 1).")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("pipeline_runs/20261125T111027.671691Z/sft.jsonl"),
        help="Source prompt-completion JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length", type=int, default=1024,
        help="Maximum tokens per training sequence (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="LoRA adapter rank (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.max_length < 1:
        parser.error("--max-length must be a positive integer.")
    if args.lora_rank < 1:
        parser.error("--lora-rank must be a positive integer.")
    if not args.dataset_path.is_file():
        parser.error(f"Dataset file does not exist: {args.dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )
    try:
        source_indices = _find_longest_source_indices(args.dataset_path, tokenizer, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path("memory_profile_runs") / run_id
    measure = profile_training_memory(
        dataset_path=args.dataset_path,
        source_indices=source_indices,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        sft_config_kwargs={
            "per_device_train_batch_size": args.batch_size,
            "max_length": args.max_length,
            "gradient_accumulation_steps": 1,
            "bf16": True,
            "optim": "adamw_torch",
        },
        lora_config_kwargs={
            "task_type": TaskType.CAUSAL_LM,
            "r": args.lora_rank,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
        output_path=run_dir / "memory_profile.json",
        generate_html=True
    )
    print(measure)


if __name__ == "__main__":
    main()
