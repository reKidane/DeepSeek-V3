"""Offline tests for the Amharic pipeline (no downloads needed).

Run from training/amharic:  python -m pytest tests -q
The training smoke test builds a tiny random model and tokenizer locally
and runs a few CPU steps of both stages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import amharic_text as at  # noqa: E402
import build_pairs as bp  # noqa: E402

AM_PARA = ("ጤፍ በኢትዮጵያ ደጋማ አካባቢዎች የሚበቅል ጠቃሚ ሰብል ነው። እንጀራ ከጤፍ ዱቄት ይጋገራል። "
           "ገበሬዎች ጤፍን በክረምት ወቅት ይዘራሉ። ")


def test_ethiopic_ratio_and_legacy_font_detection():
    assert at.ethiopic_ratio("ሰላም ዓለም") == 1.0
    assert at.ethiopic_ratio("hello") == 0.0
    assert at.looks_like_unicode_amharic(AM_PARA)
    assert not at.looks_like_unicode_amharic("ÃxÂU yxS ¨ÃU")  # legacy-font style output


def test_normalize_punctuation_and_whitespace():
    assert at.normalize("ሰላም  ነው ::") == "ሰላም ነው ።"
    assert at.normalize("ሀ\r\n\r\n\r\nለ") == "ሀ\n\nለ"


def test_homophone_folding_and_dedup_key():
    assert at.fold_homophones("ሐሠዐፀ") == "ሀሰአጸ"
    assert at.fold_homophones("ኀ") == "ሀ"
    assert at.dedup_key("ፀሐይ ወጣች።") == at.dedup_key("ጸሀይ  ወጣች")
    assert at.dedup_key("ፀሐይ") != at.dedup_key("ጨረቃ")


def test_split_sentences_and_word_count():
    sents = at.split_sentences(AM_PARA)
    assert len(sents) == 3 and sents[0].endswith("።")
    assert at.word_count("ሰላም ዓለም፡ነው") == 3


def test_dictionary_pairs():
    rows = [{"word": "ጥበብ", "definition": "ዕውቀትና ብልሃት", "english": "wisdom",
             "pos": "ስም", "example": "ጥበብ ከወርቅ ትበልጣለች።"},
            {"word": "ውሃ", "definition": "", "english": "water", "pos": "", "example": ""},
            {"word": "", "definition": "ignored", "english": "", "pos": "", "example": ""}]
    pairs = bp.dictionary_pairs(rows)
    # word 1: define + sentence + en->am + am->en ; word 2: define + en->am + am->en
    assert len(pairs) == 7
    first = pairs[0]["messages"]
    assert first[0]["content"] == "'ጥበብ' የሚለው ቃል ትርጉሙ ምንድን ነው?"
    assert "ትርጉም፦ ዕውቀትና ብልሃት" in first[1]["content"]
    assert all(p["source"] == "dictionary" for p in pairs)


def test_conjugation_pairs_include_table():
    rows = [{"verb": "መብላት", "tense": "ኃላፊ", "person": p, "form": f}
            for p, f in [("እኔ", "በላሁ"), ("አንተ", "በላህ"), ("እሷ", "በላች")]]
    pairs = bp.conjugation_pairs(rows)
    assert len(pairs) == 4
    assert pairs[0]["messages"][0]["content"] == "'መብላት' የሚለውን ግስ በኃላፊ ጊዜ ለ'እኔ' አርባው።"
    assert pairs[0]["messages"][1]["content"] == "በላሁ"
    assert "አንተ፦ በላህ" in pairs[-1]["messages"][1]["content"]


def test_text_chunks_filter_and_split():
    text = "\n\n".join([AM_PARA * 3] * 4) + "\n\nThis English paragraph is long enough " * 10
    chunks = bp.text_chunks(text, max_chars=600, min_chars=100)
    assert chunks and all(len(c) <= 600 for c in chunks)
    assert all(at.looks_like_unicode_amharic(c) for c in chunks)


def test_csv_reader_handles_tsv(tmp_path):
    p = tmp_path / "dict.tsv"
    p.write_text("word\tenglish\nውሃ\twater\n", encoding="utf-8")
    rows = bp._read_csv(p)
    assert rows == [{"word": "ውሃ", "english": "water"}]


def test_build_pairs_cli(tmp_path):
    d = tmp_path / "d.csv"
    d.write_text("word,definition,english\nጥበብ,ዕውቀትና ብልሃት,wisdom\nጥበብ,ዕውቀትና ብልሃት,wisdom\n",
                 encoding="utf-8")
    g = tmp_path / "g.txt"
    g.write_text((AM_PARA * 5 + "\n\n") * 3, encoding="utf-8")
    out = tmp_path / "out"
    assert bp.main(["--dictionary", str(d), "--grammar-text", str(g), "--out", str(out)]) == 0
    sft = [json.loads(l) for l in (out / "sft_sources.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(sft) == 3  # duplicate row removed
    assert (out / "pretrain_sources.jsonl").read_text(encoding="utf-8").strip()


def test_extract_text_txt_and_docx(tmp_path):
    import extract_text as et
    from docx import Document

    t = tmp_path / "a.txt"
    t.write_text(AM_PARA, encoding="utf-8")
    doc = Document()
    doc.add_paragraph(AM_PARA)
    dx = tmp_path / "b.docx"
    doc.save(str(dx))
    assert et.main([str(t), str(dx), "--out-dir", str(tmp_path / "x")]) == 0
    assert (tmp_path / "x" / "b.txt").read_text(encoding="utf-8").startswith("ጤፍ")


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    corpus = [AM_PARA, "A B C D Answer with the letter", "ሰላም ጥበብ መብላት በላሁ"] * 20
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<unk>", "<pad>", "<end>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                                  eos_token="<end>")
    tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<end>{% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=16, max_position_embeddings=256,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return Qwen3ForCausalLM(cfg), tok


def test_training_smoke_both_stages(tmp_path, tiny_model_and_tokenizer):
    import copy

    import train_lora as tl

    model, tok = tiny_model_and_tokenizer
    pre = tmp_path / "pretrain.jsonl"
    pre.write_text("\n".join(json.dumps({"text": AM_PARA * 2}, ensure_ascii=False) for _ in range(8)),
                   encoding="utf-8")
    sft = tmp_path / "sft.jsonl"
    rec = {"messages": [{"role": "user", "content": "ጥበብ ምንድን ነው?"},
                        {"role": "assistant", "content": "ዕውቀትና ብልሃት"}]}
    sft.write_text("\n".join(json.dumps(rec, ensure_ascii=False) for _ in range(8)), encoding="utf-8")

    a = tl.train("pretrain", "tiny", pre, tmp_path / "a", max_steps=2, batch_size=2, grad_accum=1,
                 max_length=64, tokenizer=tok, model_obj=copy.deepcopy(model), save_steps=1000)
    assert (a / "adapter_config.json").exists()
    b = tl.train("sft", "tiny", sft, tmp_path / "b", resume_adapter=a, eval_data=sft, max_steps=2,
                 batch_size=2, grad_accum=1, max_length=64, tokenizer=tok,
                 model_obj=copy.deepcopy(model), save_steps=1)
    assert (b / "adapter_config.json").exists()
    assert json.loads((b / "train_log.json").read_text())


def test_eval_functions_on_tiny_model(tiny_model_and_tokenizer):
    import eval_amharic as ev

    model, tok = tiny_model_and_tokenizer
    items = [{"context": AM_PARA, "question": "ጤፍ ምንድን ነው?",
              "choices": ["ሰብል", "እንስሳ", "ወንዝ", "ተራራ"], "answer": 0}] * 3
    res = ev.score_mc(model, tok, items, batch_size=2)
    assert res["n"] == 3 and 0.0 <= res["accuracy"] <= 1.0
    fert = ev.fertility(tok, [AM_PARA])
    assert fert["tokens_per_word"] > 0 and fert["words"] == at.word_count(AM_PARA)
    out = ev.run_prompts(model, tok, max_new_tokens=3)
    assert len(out) == len(ev.BASELINE_PROMPTS) and "answer" in out[0]


def test_build_corpus_with_fake_hub(tmp_path, monkeypatch):
    """Runs build_corpus end to end on small local stand-ins for the Hub files."""
    import pandas as pd

    import build_corpus as bc

    leaked = "ይህ ዓረፍተ ነገር ከግምገማው ስብስብ የተወሰደ ስለሆነ መወገድ አለበት።"
    wiki = tmp_path / "wiki.parquet"
    pd.DataFrame({"title": ["ጤፍ"], "text": [(AM_PARA * 6 + "\n\n") * 2]}).to_parquet(wiki)
    web = tmp_path / "web.parquet"
    pd.DataFrame({"text": [AM_PARA * 5 + leaked, "ንጹህ " + AM_PARA * 6]}).to_parquet(web)
    aya = tmp_path / "aya.parquet"
    pd.DataFrame({"inputs": ["ጤፍ ምንድን ነው?", "What is teff?"],
                  "targets": ["ጤፍ የእህል ዓይነት ነው።", "A grain."],
                  "language": ["Amharic", "English"],
                  "language_code": ["amh", "eng"]}).to_parquet(aya)
    bel = tmp_path / "bel.jsonl"
    bel.write_text(json.dumps({"flores_passage": leaked + " ሌላ ዓረፍተ ነገር ደግሞ እዚህ አለ።"},
                              ensure_ascii=False) + "\n", encoding="utf-8")
    mmlu = tmp_path / "mmlu.tsv"
    mmlu.write_text("question\tchoices\tanswer\tsubject\nጥያቄ?\t['a','b','c','d']\tA\tx\n",
                    encoding="utf-8")
    files = {bc.WIKI: wiki, bc.FINEWEB: web, bc.AYA: aya, bc.BELEBELE: bel, bc.AFRIMMLU: mmlu}
    monkeypatch.setattr(bc, "hub_file", lambda repo, fn: str(files[(repo, fn)]))

    out = tmp_path / "data"
    assert bc.main(["--out", str(out), "--max-web-docs", "10"]) == 0
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    texts = (out / "pretrain.jsonl").read_text(encoding="utf-8")
    assert leaked not in texts  # decontaminated
    assert stats["pretrain_by_source"].get("wikipedia") and stats["pretrain_by_source"].get("fineweb2")
    assert stats["sft_by_source"] == {"aya": 1}
