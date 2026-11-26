"""Extract plain text from grammar or dictionary source files.

Supports .txt/.md, .docx and .pdf. Reports whether the result is real
Unicode Amharic: PDFs typed with legacy Amharic fonts (for example older
Power Geez or Visual Geez documents) extract as Latin gibberish and must be
converted or re-typed before they are usable.

Usage:
    python extract_text.py SOURCE [SOURCE ...] --out-dir extracted/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from amharic_text import ethiopic_ratio, normalize


def extract(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append("\t".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("extracted"))
    ap.add_argument("--min-ratio", type=float, default=0.5,
                    help="minimum share of Ethiopic letters to accept a file")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for src in args.sources:
        text = normalize(extract(src))
        ratio = ethiopic_ratio(text)
        ok = ratio >= args.min_ratio
        out = args.out_dir / (src.stem + ".txt")
        if ok:
            out.write_text(text, encoding="utf-8")
        report.append({"source": str(src), "chars": len(text),
                       "ethiopic_ratio": round(ratio, 3), "accepted": ok})
        status = "OK" if ok else "REJECTED (not Unicode Amharic; legacy font?)"
        print(f"{src.name}: {len(text):,} chars, Ethiopic {ratio:.0%} -> {status}")
    (args.out_dir / "extract_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["accepted"] for r in report) else 1


if __name__ == "__main__":
    sys.exit(main())
