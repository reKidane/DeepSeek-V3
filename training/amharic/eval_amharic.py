"""Evaluate Amharic ability of a model (optionally with a LoRA adapter).

Measures:
  fertility   tokens per Amharic word for one or more tokenizers (lower is
              more efficient: faster and cheaper to run)
  belebele    reading comprehension, 4-choice (facebook/belebele amh_Ethi)
  afrimmlu    knowledge questions, 4-choice (masakhane/afrimmlu amh test)
  prompts     free-form answers to the Tibeb baseline prompts, saved for
              review by a native speaker

Multiple-choice items are scored by comparing the model's next-token
probabilities for the answer letters A-D, so no answer parsing is needed.

Usage:
    python eval_amharic.py --model Qwen/Qwen3-4B-Instruct-2507 --out results/base.json
    python eval_amharic.py --model Qwen/Qwen3-4B-Instruct-2507 --adapter out/lora --out results/tuned.json
    python eval_amharic.py --fertility-only --tokenizers Qwen/Qwen3-4B-Instruct-2507 deepseek-ai/DeepSeek-V4-Pro-0813
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from amharic_text import word_count

BELEBELE = ("facebook/belebele", "data/amh_Ethi.jsonl")
AFRIMMLU = ("masakhane/afrimmlu", "data/amh/test.tsv")
LETTERS = ["A", "B", "C", "D"]

TIBEB_SYSTEM = ('You are Tibeb, an AI assistant. Your name means "wisdom" in Amharic. '
                "You are built on the open DeepSeek-V4 model and maintained by Kidanemaryam W. Reta.")

BASELINE_PROMPTS = [
    {"id": 1, "prompt": "Who are you and who maintains you? Answer in English in two sentences."},
    {"id": 2, "prompt": "ሰላም! ስለ ራስህ በአጭሩ በአማርኛ ንገረኝ።"},
    {"id": 3, "prompt": "'መብላት' የሚለውን ግስ በኃላፊ ጊዜ ለ'እኔ'፣ 'አንተ'፣ 'አንቺ'፣ 'እሱ' እና 'እሷ' አርባው።",
     "reference": "በላሁ, በላህ, በላሽ, በላ, በላች"},
    {"id": 4, "prompt": 'Translate into Amharic: "Drought reduces sorghum yield in the Ethiopian highlands."'},
    {"id": 5, "prompt": "ጤፍ በኢትዮጵያ ለምን አስፈላጊ ሰብል ነው? በአማርኛ በሶስት ዓረፍተ ነገሮች መልስ።"},
    {"id": 6, "prompt": "'ጥበብ' የሚለው ቃል ትርጉሙ ምንድን ነው? ሁለት ምሳሌ ዓረፍተ ነገሮች ስጠኝ።"},
    {"id": 7, "prompt": "'ቀስ በቀስ እንቁላል በእግሩ ይሄዳል' የሚለውን ተረት ትርጉም አስረዳ።"},
]


def hub_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def load_belebele(limit: int | None = None) -> list[dict]:
    with open(hub_file(*BELEBELE), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    items = [{"context": r["flores_passage"], "question": r["question"],
              "choices": [r[f"mc_answer{i}"] for i in range(1, 5)],
              "answer": int(r["correct_answer_num"]) - 1} for r in rows]
    return items[:limit] if limit else items


def load_afrimmlu(limit: int | None = None) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(hub_file(*AFRIMMLU), sep="\t")
    items = []
    for q, ch, a in zip(df["question"], df["choices"], df["answer"]):
        choices = ast.literal_eval(ch) if isinstance(ch, str) else list(ch)
        if len(choices) == 4 and str(a).strip() in LETTERS:
            items.append({"context": "", "question": str(q), "choices": choices,
                          "answer": LETTERS.index(str(a).strip())})
    return items[:limit] if limit else items


def fertility(tokenizer, texts: list[str]) -> dict:
    words = sum(word_count(t) for t in texts)
    chars = sum(len(t) for t in texts)
    tokens = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts)
    return {"tokens_per_word": round(tokens / max(words, 1), 3),
            "chars_per_token": round(chars / max(tokens, 1), 3), "words": words}


def mc_prompt(item: dict) -> str:
    parts = []
    if item["context"]:
        parts.append(f"Passage:\n{item['context']}\n")
    parts.append(f"Question: {item['question']}")
    parts += [f"{L}. {c}" for L, c in zip(LETTERS, item["choices"])]
    parts.append("Answer with the letter of the correct option only.")
    return "\n".join(parts)


def chat_text(tokenizer, user: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def letter_ids(tokenizer) -> list[int]:
    ids = []
    for L in LETTERS:
        toks = tokenizer(L, add_special_tokens=False)["input_ids"]
        ids.append(toks[0])
    return ids


def score_mc(model, tokenizer, items: list[dict], batch_size: int = 4) -> dict:
    import torch

    ids = letter_ids(tokenizer)
    correct = 0
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        texts = [chat_text(tokenizer, mc_prompt(it)) for it in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        preds = logits[:, ids].argmax(dim=-1).tolist()
        correct += sum(int(p == it["answer"]) for p, it in zip(preds, batch))
    return {"accuracy": round(correct / max(len(items), 1), 4), "n": len(items)}


def run_prompts(model, tokenizer, max_new_tokens: int = 400) -> list[dict]:
    import torch

    out = []
    for p in BASELINE_PROMPTS:
        text = chat_text(tokenizer, p["prompt"], TIBEB_SYSTEM)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        answer = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({**p, "answer": answer.strip()})
    return out


def load_model(model_id: str, adapter: str | None = None, four_bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter or model_id)
    kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.float16
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def evaluate(model, tokenizer, limit: int | None = None, prompts: bool = True) -> dict:
    texts = list({it["context"] for it in load_belebele()})
    res = {"fertility": fertility(tokenizer, texts),
           "belebele": score_mc(model, tokenizer, load_belebele(limit)),
           "afrimmlu": score_mc(model, tokenizer, load_afrimmlu(limit))}
    if prompts:
        res["prompts"] = run_prompts(model, tokenizer)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--limit", type=int, default=None, help="max items per benchmark")
    ap.add_argument("--no-prompts", action="store_true")
    ap.add_argument("--fertility-only", action="store_true")
    ap.add_argument("--tokenizers", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("results/eval.json"))
    args = ap.parse_args(argv)

    if args.fertility_only:
        from transformers import AutoTokenizer

        texts = list({it["context"] for it in load_belebele()})
        res = {}
        for t in args.tokenizers:
            try:
                res[t] = fertility(AutoTokenizer.from_pretrained(t), texts)
            except Exception as err:  # gated, missing or unsupported tokenizer
                res[t] = {"error": f"{type(err).__name__}: {err}"[:200]}
    else:
        model, tok = load_model(args.model, args.adapter)
        res = {"model": args.model, "adapter": args.adapter,
               **evaluate(model, tok, args.limit, not args.no_prompts)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "prompts"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
