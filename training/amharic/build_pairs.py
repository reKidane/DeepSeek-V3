"""Turn Amharic dictionary and grammar sources into training records.

Inputs (all optional, any combination):
  --dictionary FILE.csv   columns: word, definition, english, pos, example
                          ("word" plus "definition" or "english" required)
  --conjugations FILE.csv columns: verb, tense, person, form
                          (tense and person written in Amharic, e.g. ኃላፊ, እኔ)
  --qa FILE.csv           columns: question, answer (hand-written grammar Q&A)
  --grammar-text FILE.txt plain Unicode text of grammar books/notes
                          (use extract_text.py first for PDF/DOCX)

Outputs (JSON Lines):
  OUT/sft_sources.jsonl       {"messages": [...], "source": ...}
  OUT/pretrain_sources.jsonl  {"text": ..., "source": ...}

The question templates are in TEMPLATES below. They are written in plain
Amharic and should be reviewed by a native speaker before large runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from amharic_text import dedup_key, looks_like_unicode_amharic, normalize

TEMPLATES = {
    "define": "'{word}' የሚለው ቃል ትርጉሙ ምንድን ነው?",
    "use_in_sentence": "'{word}' የሚለውን ቃል በዓረፍተ ነገር ውስጥ ተጠቀምበት።",
    "en_to_am": 'What is the Amharic word for "{english}"?',
    "am_to_en": 'What does the Amharic word "{word}" mean in English?',
    "conjugate_one": "'{verb}' የሚለውን ግስ በ{tense} ጊዜ ለ'{person}' አርባው።",
    "conjugate_all": "'{verb}' የሚለውን ግስ በ{tense} ጊዜ አርባው።",
}
LABELS = {"word": "ቃል", "pos": "የንግግር ክፍል", "meaning": "ትርጉም",
          "english": "በእንግሊዝኛ", "example": "ምሳሌ"}


def _chat(user: str, assistant: str, source: str) -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}],
            "source": source}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;") if sample else csv.excel
        rows = list(csv.DictReader(f, dialect=dialect))
    return [{(k or "").strip().lower(): normalize(v or "") for k, v in r.items()}
            for r in rows]


def dictionary_pairs(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        word, definition = r.get("word", ""), r.get("definition", "")
        english, pos, example = r.get("english", ""), r.get("pos", ""), r.get("example", "")
        if not word or not (definition or english):
            continue
        lines = [f"{LABELS['word']}፦ {word}"]
        if pos:
            lines.append(f"{LABELS['pos']}፦ {pos}")
        if definition:
            lines.append(f"{LABELS['meaning']}፦ {definition}")
        if english:
            lines.append(f"{LABELS['english']}፦ {english}")
        if example:
            lines.append(f"{LABELS['example']}፦ {example}")
        out.append(_chat(TEMPLATES["define"].format(word=word), "\n".join(lines), "dictionary"))
        if example:
            out.append(_chat(TEMPLATES["use_in_sentence"].format(word=word), example, "dictionary"))
        if english:
            out.append(_chat(TEMPLATES["en_to_am"].format(english=english),
                             f'"{english}" in Amharic is {word}.', "dictionary"))
            out.append(_chat(TEMPLATES["am_to_en"].format(word=word),
                             f'"{word}" means "{english}".', "dictionary"))
    return out


def conjugation_pairs(rows: list[dict]) -> list[dict]:
    out = []
    tables: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for r in rows:
        verb, tense, person, form = (r.get(k, "") for k in ("verb", "tense", "person", "form"))
        if not all((verb, tense, person, form)):
            continue
        out.append(_chat(TEMPLATES["conjugate_one"].format(verb=verb, tense=tense, person=person),
                         form, "conjugation"))
        tables[(verb, tense)].append((person, form))
    for (verb, tense), forms in tables.items():
        if len(forms) > 1:
            answer = "\n".join(f"{p}፦ {f}" for p, f in forms)
            out.append(_chat(TEMPLATES["conjugate_all"].format(verb=verb, tense=tense),
                             answer, "conjugation"))
    return out


def qa_pairs(rows: list[dict]) -> list[dict]:
    return [_chat(r["question"], r["answer"], "grammar_qa")
            for r in rows if r.get("question") and r.get("answer")]


def text_chunks(text: str, max_chars: int = 1500, min_chars: int = 200) -> list[str]:
    """Split text into chunks on paragraph boundaries."""
    chunks, buf = [], ""
    for para in normalize(text).split("\n\n"):
        if buf and len(buf) + len(para) + 2 > max_chars:
            chunks.append(buf)
            buf = ""
        while len(para) > max_chars:  # very long paragraph
            cut = para.rfind("።", 0, max_chars)
            cut = cut + 1 if cut > min_chars else max_chars
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= min_chars and looks_like_unicode_amharic(c)]


def dedupe(records: list[dict], key_fn) -> list[dict]:
    seen, out = set(), []
    for r in records:
        k = dedup_key(key_fn(r))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dictionary", type=Path, nargs="*", default=[])
    ap.add_argument("--conjugations", type=Path, nargs="*", default=[])
    ap.add_argument("--qa", type=Path, nargs="*", default=[])
    ap.add_argument("--grammar-text", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args(argv)

    sft: list[dict] = []
    for p in args.dictionary:
        sft += dictionary_pairs(_read_csv(p))
    for p in args.conjugations:
        sft += conjugation_pairs(_read_csv(p))
    for p in args.qa:
        sft += qa_pairs(_read_csv(p))
    pretrain = [{"text": c, "source": f"grammar:{p.name}"}
                for p in args.grammar_text
                for c in text_chunks(p.read_text(encoding="utf-8"))]

    sft = dedupe(sft, lambda r: r["messages"][0]["content"] + r["messages"][1]["content"])
    pretrain = dedupe(pretrain, lambda r: r["text"])

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "sft_sources.jsonl", sft)
    write_jsonl(args.out / "pretrain_sources.jsonl", pretrain)
    print(f"SFT pairs: {len(sft):,}  pretraining chunks: {len(pretrain):,}  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
