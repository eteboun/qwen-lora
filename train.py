import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from project_config import MODEL_ID, MODEL_REVISION


DEFAULT_DATASET_PATH = Path("pipeline_runs/20261125T111027.671691Z/sft.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on the prepared Python SFT dataset.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--smoke-test", action="store_true", help="Run 10 optimizer steps instead of one epoch.")
    args = parser.parse_args()
    if not args.dataset_path.is_file():
        parser.error(f"Dataset file does not exist: {args.dataset_path}")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access CUDA.")

    dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    validation_size = 500
    split_seed = 42
    if len(dataset) <= validation_size:
        parser.error(f"The dataset must contain more than {validation_size} rows for the validation split.")
    splits = dataset.train_test_split(test_size=validation_size, seed=split_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=0,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path("training_runs") / run_id
    sft_config = SFTConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=10 if args.smoke_test else -1,
        max_length=1024,
        completion_only_loss=True,
        bf16=True,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="no",
        logging_first_step=True,
        logging_steps=10,
        report_to="none",
        seed=42,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        peft_config=lora_config,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
    )

    lora_settings = trainer.model.active_peft_config.to_dict()
    for key, value in lora_settings.items():
        if isinstance(value, set):
            lora_settings[key] = sorted(value)
    run_config = {
        "dataset_path": str(args.dataset_path.resolve()),
        "split_seed": split_seed,
        "train_size": len(splits["train"]),
        "validation_size": validation_size,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": str(model.dtype),
        "sft_config": trainer.args.to_dict(),
        "lora_config": lora_settings,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    trainer.train()
    trainer.save_state()
    adapter_path = run_dir / "adapter"
    trainer.save_model(str(adapter_path))
    print(f"Saved LoRA adapter to {adapter_path}")


if __name__ == "__main__":
    main()
