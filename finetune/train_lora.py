"""LoRA / QLoRA-SFT training.

``dry_run`` validates the config + dataset off-GPU (no torch import). ``train`` runs the real
job and lazily imports the CUDA stack — install ``requirements/train.txt`` on the Blackwell
box (cu128) first. Defaults come from ``configs/lora.yaml`` (RTX 5090 -> QLoRA 4-bit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.config import load_yaml

DEFAULT_CONFIG = "configs/lora.yaml"
DEFAULT_DATA = "outputs/sft/zh_service_desk.jsonl"


def _count_jsonl(path: str) -> int:
    file = Path(path)
    if not file.exists():
        return 0
    return sum(1 for line in file.read_text(encoding="utf-8").splitlines() if line.strip())


def dry_run(config_path: str = DEFAULT_CONFIG, data_path: str = DEFAULT_DATA) -> dict[str, Any]:
    """Validate the training config + dataset without importing torch."""
    cfg = load_yaml(config_path)
    samples = _count_jsonl(data_path)
    return {
        "method": cfg.get("method"),
        "base_model": cfg.get("base_model"),
        "lora": cfg.get("lora"),
        "num_train_epochs": cfg.get("train", {}).get("num_train_epochs"),
        "load_in_4bit": cfg.get("quantization", {}).get("load_in_4bit"),
        "samples": samples,
        "ready": samples > 0,
    }


def train(config_path: str = DEFAULT_CONFIG, data_path: str = DEFAULT_DATA) -> None:
    """Run LoRA/QLoRA-SFT on the GPU box (lazy CUDA imports; saves the adapter only)."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - exercised only on the GPU box
        raise RuntimeError(
            "Training needs requirements/train.txt on the Blackwell GPU box "
            "(torch from the cu128 index). See README."
        ) from exc

    cfg = load_yaml(config_path)
    lora_cfg = cfg["lora"]
    train_cfg = cfg["train"]
    quant = cfg.get("quantization", {})

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if cfg.get("method") == "qlora" and quant.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **model_kwargs)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    dataset = load_dataset("json", data_files=data_path, split="train")

    def _format(example: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": text}

    dataset = dataset.map(_format)
    sft_config = SFTConfig(
        output_dir=cfg["output"]["adapter_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_seq_length=train_cfg["max_seq_len"],
        bf16=train_cfg["bf16"],
        seed=train_cfg["seed"],
    )
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, peft_config=peft_config)
    trainer.train()
    trainer.save_model(cfg["output"]["adapter_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA-SFT for PolicyArena.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--dry-run", action="store_true", help="validate config+data only (no GPU)")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run(args.config, args.data), ensure_ascii=False, indent=2))
    else:
        train(args.config, args.data)


if __name__ == "__main__":
    main()
