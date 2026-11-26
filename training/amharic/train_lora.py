"""LoRA / QLoRA training for Amharic on a small open model.

Two stages, run one after the other:
  pretrain  continued training on plain Amharic text (pretrain.jsonl)
  sft       instruction tuning on question/answer pairs (sft.jsonl);
            loss is computed on the answers only

On a CUDA GPU the base model is loaded in 4-bit (QLoRA) with fp16 compute,
which fits a 4B model on one free Kaggle T4 (16 GB). On CPU it runs in full
precision, which is only useful for smoke tests with tiny models.

Usage:
    python train_lora.py --stage pretrain --model Qwen/Qwen3-4B-Instruct-2507 \
        --data data/pretrain.jsonl --out out/stage_a --max-steps 1500
    python train_lora.py --stage sft --model Qwen/Qwen3-4B-Instruct-2507 \
        --data data/sft.jsonl --eval-data data/sft_eval.jsonl \
        --resume-adapter out/stage_a --out out/stage_b --epochs 1
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


def load_jsonl_dataset(path: Path, stage: str, limit: int | None = None):
    from datasets import Dataset

    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if stage == "pretrain":
                rows.append({"text": r["text"]})
            else:
                msgs = r["messages"]
                rows.append({"prompt": msgs[:-1], "completion": msgs[-1:]})
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def load_base(model_id: str, model_obj=None):
    import torch
    from transformers import AutoModelForCausalLM

    if model_obj is not None:
        return model_obj
    if torch.cuda.is_available():
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0}, dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return AutoModelForCausalLM.from_pretrained(model_id)


def train(stage: str, model_id: str, data: Path, out: Path, *, eval_data: Path | None = None,
          resume_adapter: Path | None = None, epochs: float = 1.0, max_steps: int = -1,
          lr: float | None = None, max_length: int = 1024, batch_size: int = 2,
          grad_accum: int = 8, lora_r: int = 16, limit: int | None = None,
          tokenizer=None, model_obj=None, save_steps: int = 200) -> Path:
    """Run one training stage and save the LoRA adapter to `out`."""
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if stage not in {"pretrain", "sft"}:
        raise ValueError("stage must be 'pretrain' or 'sft'")
    tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_base(model_id, model_obj)
    peft_config = None
    if resume_adapter:
        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
    else:
        peft_config = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    train_ds = load_jsonl_dataset(data, stage, limit)
    eval_ds = load_jsonl_dataset(eval_data, stage) if eval_data and eval_data.exists() else None
    cuda = torch.cuda.is_available()
    # transformers 5 folded warmup_ratio into warmup_steps (values < 1 are a ratio)
    params = inspect.signature(SFTConfig.__init__).parameters
    warmup = {"warmup_ratio": 0.03} if "warmup_ratio" in params else {"warmup_steps": 0.03}
    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr or (1e-4 if stage == "pretrain" else 2e-4),
        lr_scheduler_type="cosine",
        **warmup,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=cuda,
        fp16=cuda,
        bf16=False,
        use_cpu=not cuda,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=save_steps if eval_ds is not None else None,
        report_to="none",
        seed=0,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--eval-data", type=Path)
    ap.add_argument("--resume-adapter", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    train(args.stage, args.model, args.data, args.out, eval_data=args.eval_data,
          resume_adapter=args.resume_adapter, epochs=args.epochs, max_steps=args.max_steps,
          lr=args.lr, max_length=args.max_length, batch_size=args.batch_size,
          grad_accum=args.grad_accum, lora_r=args.lora_r, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
