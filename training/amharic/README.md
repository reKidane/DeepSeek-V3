# Tibeb Amharic lab

A free, reproducible pipeline for improving a model's Amharic. It is developed on a small open model using free Kaggle GPUs, and the same recipe is intended for Tibeb-Pro once larger compute is available.

## Quick start on Kaggle (free)

1. Open [kaggle.com](https://www.kaggle.com), create a notebook, then **File → Import Notebook** and upload `kaggle_amharic_qlora.ipynb` (or import it from this GitHub repo).
2. Settings: **GPU T4 x2** (or P100) and **Internet on**.
3. Optional: attach your own sources as a Kaggle dataset named `amharic-sources` (formats below).
4. Optional: add an `HF_TOKEN` secret to upload the trained adapter to Hugging Face.
5. **Run all.** The notebook reports tokenizer efficiency, baseline scores, trains in two stages, re-scores, and prints base vs. trained answers side by side for your review.

## What it does

| Step | Script | Output |
| :--- | :--- | :--- |
| Extract text from your grammar/dictionary files | `extract_text.py` | `extracted/*.txt` plus a report flagging non-Unicode (legacy-font) files |
| Turn your sources into training records | `build_pairs.py` | `data/sft_sources.jsonl`, `data/pretrain_sources.jsonl` |
| Add public Amharic data, deduplicate, remove eval overlap | `build_corpus.py` | `data/pretrain.jsonl`, `data/sft.jsonl`, `data/sft_eval.jsonl`, `data/stats.json` |
| Measure Amharic ability | `eval_amharic.py` | tokens per word, Belebele and AfriMMLU accuracy, baseline prompt answers |
| Train a LoRA adapter (4-bit QLoRA on GPU) | `train_lora.py` | `out/stage_a` (continued training), `out/stage_b` (instruction tuning) |

Default base model: `Qwen/Qwen3-4B-Instruct-2507` (text-only, Apache-2.0, fits one T4 in 4-bit). Change `BASE_MODEL` in the notebook to try others.

## Your own sources

Put these in the `amharic-sources` dataset. Files are matched by name prefix; see `templates/` for examples.

| File | Columns | Becomes |
| :--- | :--- | :--- |
| `dictionary*.csv` | `word, definition, english, pos, example` (word plus definition or english required) | definition, usage, and Amharic↔English questions |
| `conjugations*.csv` | `verb, tense, person, form` (tense and person in Amharic, e.g. ኃላፊ, እኔ) | single-form and full-table conjugation questions |
| `qa*.csv` | `question, answer` | used as written |
| grammar `.txt`, `.docx`, `.pdf` | Unicode Amharic text | continued-training text |

Notes:

- **Legacy fonts:** PDFs typed with older non-Unicode Amharic fonts extract as gibberish and are rejected automatically. Convert them to Unicode first or re-type key sections.
- **Templates:** the Amharic question wording is in `TEMPLATES` at the top of `build_pairs.py`; review and edit it before large runs.
- **Rights:** only train on material you are allowed to use. Published grammar books and dictionaries are usually copyrighted, so check permission before releasing a model trained on them.

## Public data used

| Source | Use | License |
| :--- | :--- | :--- |
| Amharic Wikipedia (`wikimedia/wikipedia`, `20231101.am`) | continued training | CC BY-SA 3.0 |
| FineWeb-2 Amharic (`HuggingFaceFW/fineweb-2`, `amh_Ethi`) | continued training | ODC-By |
| Aya dataset, Amharic rows (`CohereLabs/aya_dataset`) | instruction tuning | Apache-2.0 |
| Belebele Amharic (`facebook/belebele`) | evaluation only | CC BY-SA 4.0 |
| AfriMMLU Amharic (`masakhane/afrimmlu`) | evaluation only | Apache-2.0 |

## Running locally

```bash
pip install -r requirements.txt
python -m pytest tests -q        # offline tests, including a tiny CPU training run
```

A Mac with 32 to 64 GB can run the data scripts and evaluate small models; training at useful speed needs a CUDA GPU (Kaggle is enough for 4B models).
