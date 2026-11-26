"""Build the Amharic training corpus from public data plus your own sources.

Public data (downloaded from the Hugging Face Hub):
  - Amharic Wikipedia        wikimedia/wikipedia, 20231101.am      (CC BY-SA 3.0)
  - FineWeb-2 Amharic web    HuggingFaceFW/fineweb-2, amh_Ethi      (ODC-By)
  - Aya Amharic instructions CohereLabs/aya_dataset, Amharic rows  (Apache 2.0)

Your own records from build_pairs.py are merged in. Text that overlaps the
evaluation sets (Belebele and AfriMMLU Amharic) is removed so evaluation
scores are not inflated.

Outputs in OUT/: pretrain.jsonl, sft.jsonl, stats.json

Usage:
    python build_corpus.py --out data --max-web-docs 50000 \
        --extra-sft data/sft_sources.jsonl --extra-pretrain data/pretrain_sources.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from amharic_text import dedup_key, looks_like_unicode_amharic, normalize, split_sentences
from build_pairs import dedupe, text_chunks, write_jsonl

WIKI = ("wikimedia/wikipedia", "20231101.am/train-00000-of-00001.parquet")
FINEWEB = ("HuggingFaceFW/fineweb-2", "data/amh_Ethi/train/000_00000.parquet")
AYA = ("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet")
BELEBELE = ("facebook/belebele", "data/amh_Ethi.jsonl")
AFRIMMLU = ("masakhane/afrimmlu", "data/amh/test.tsv")


def hub_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_sentence_keys() -> set[str]:
    """Sentence-level keys from the evaluation sets, for decontamination."""
    import pandas as pd

    keys: set[str] = set()
    for row in read_jsonl(Path(hub_file(*BELEBELE))):
        for s in split_sentences(row["flores_passage"]):
            if len(s) > 20:
                keys.add(dedup_key(s))
    mmlu = pd.read_csv(hub_file(*AFRIMMLU), sep="\t")
    for q in mmlu["question"].astype(str):
        if len(q) > 20:
            keys.add(dedup_key(q))
    return keys


def is_contaminated(text: str, keys: set[str]) -> bool:
    return any(dedup_key(s) in keys for s in split_sentences(text) if len(s) > 20)


def wiki_chunks() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(hub_file(*WIKI), columns=["title", "text"])
    return [{"text": c, "source": "wikipedia"}
            for t in df["text"].astype(str) for c in text_chunks(t)]


def web_chunks(max_docs: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(hub_file(*FINEWEB))
    rng = random.Random(seed)
    groups = list(range(pf.num_row_groups))
    rng.shuffle(groups)
    out: list[dict] = []
    docs = 0
    for g in groups:
        for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist():
            out += [{"text": c, "source": "fineweb2"} for c in text_chunks(t)]
            docs += 1
            if docs >= max_docs:
                return out
    return out


def aya_pairs() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(hub_file(*AYA))
    lang = df.get("language_code", df.get("language")).astype(str)
    df = df[lang.str.lower().isin({"amh", "amharic"})]
    out = []
    for q, a in zip(df["inputs"].astype(str), df["targets"].astype(str)):
        q, a = normalize(q), normalize(a)
        if q and a and looks_like_unicode_amharic(q + a, 0.3):
            out.append({"messages": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a}],
                        "source": "aya"})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--max-web-docs", type=int, default=50_000)
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--no-aya", action="store_true")
    ap.add_argument("--extra-sft", type=Path, nargs="*", default=[])
    ap.add_argument("--extra-pretrain", type=Path, nargs="*", default=[])
    ap.add_argument("--sft-eval-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    pretrain: list[dict] = []
    sft: list[dict] = []
    for p in args.extra_pretrain:
        pretrain += read_jsonl(p)
    for p in args.extra_sft:
        sft += read_jsonl(p)
    if not args.no_wiki:
        pretrain += wiki_chunks()
    if not args.no_web:
        pretrain += web_chunks(args.max_web_docs, args.seed)
    if not args.no_aya:
        sft += aya_pairs()

    keys = eval_sentence_keys()
    before = len(pretrain), len(sft)
    pretrain = [r for r in dedupe(pretrain, lambda r: r["text"])
                if not is_contaminated(r["text"], keys)]
    sft = [r for r in dedupe(sft, lambda r: r["messages"][0]["content"] + r["messages"][1]["content"])
           if not is_contaminated(r["messages"][0]["content"] + "\n" + r["messages"][1]["content"], keys)]

    rng = random.Random(args.seed)
    rng.shuffle(pretrain)
    rng.shuffle(sft)
    n_eval = max(1, int(len(sft) * args.sft_eval_fraction)) if sft else 0

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "pretrain.jsonl", pretrain)
    write_jsonl(args.out / "sft.jsonl", sft[n_eval:])
    write_jsonl(args.out / "sft_eval.jsonl", sft[:n_eval])

    def by_source(recs):
        counts: dict[str, int] = {}
        for r in recs:
            counts[r["source"].split(":")[0]] = counts.get(r["source"].split(":")[0], 0) + 1
        return counts

    stats = {"pretrain_chunks": len(pretrain), "pretrain_chars": sum(len(r["text"]) for r in pretrain),
             "pretrain_by_source": by_source(pretrain), "sft_train": len(sft) - n_eval,
             "sft_eval": n_eval, "sft_by_source": by_source(sft),
             "removed_duplicates_or_eval_overlap": {"pretrain": before[0] - len(pretrain),
                                                   "sft": before[1] - len(sft)}}
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
