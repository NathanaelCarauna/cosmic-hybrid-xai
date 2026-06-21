#!/usr/bin/env python3
"""
XML SRS Documents -> COSMET format adapter.

Parses structured XML requirement documents (from XMLZIPFile_02.zip) and
extracts requirement sections into COSMET-compatible FP_prompts/*.txt.

Each <text_body> with sufficient content becomes a separate FP prompt.
Nested sections are flattened; each produces one .txt file.

No COSMIC ground truth exists for these documents.

Usage:
    python scripts/adapters/xml_to_cosmet.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from xml.etree import ElementTree as ET

BASE = Path(__file__).resolve().parent.parent.parent
XML_DIR = BASE / "datasets" / "raw" / "xml-requirements"
OUT = BASE / "datasets" / "prepared" / "xml-cosmet"

MIN_CHARS = 80
MIN_WORDS = 10


def extract_sections(xml_path: Path) -> list[dict]:
    """Extract text_body sections from an XML SRS document."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

    doc_title = ""
    title_el = root.find(f"{ns}title")
    if title_el is not None:
        titles = [t.text.strip() for t in title_el.iter(f"{ns}title") if t.text]
        if not titles and title_el.text:
            titles = [title_el.text.strip()]
        doc_title = " - ".join(titles[:2]) if titles else xml_path.stem

    sections = []
    for p_el in root.iter(f"{ns}p"):
        sec_id = p_el.get("id", "")
        sec_title_el = p_el.find(f"{ns}title")
        sec_title = sec_title_el.text.strip() if sec_title_el is not None and sec_title_el.text else ""

        for tb in p_el.findall(f"{ns}text_body"):
            if tb.text and tb.text.strip():
                text = re.sub(r"\s+", " ", tb.text.strip())
                if len(text) >= MIN_CHARS and len(text.split()) >= MIN_WORDS:
                    sections.append({
                        "doc": xml_path.stem,
                        "doc_title": doc_title,
                        "section_id": sec_id,
                        "section_title": sec_title,
                        "text": text,
                    })
    return sections


def generate_fp_prompts(all_sections: list[dict], out_dir: Path) -> None:
    fp_dir = out_dir / "FP_prompts"
    fp_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, sec in enumerate(all_sections, 1):
        short_title = sec["section_title"][:60] if sec["section_title"] else sec["text"][:60]

        lines = sec["text"].split(". ")
        scenario_lines = []
        for step, sentence in enumerate(lines, 1):
            sentence = sentence.strip().rstrip(".")
            if sentence:
                scenario_lines.append(f"{step}\tSystem\t{sentence}.")

        content = (
            f"ID\n"
            f"XML-{sec['doc']}-{sec['section_id']}\t{short_title}\n"
            f"Description\t{sec['text'][:200]}\n"
            f"Source document\t{sec['doc_title']}\n"
            f"MAIN SCENARIO\n"
            + "\n".join(scenario_lines)
            + "\n"
        )
        fp_path = fp_dir / f"FP{idx}.txt"
        fp_path.write_text(content, encoding="utf-8")

        manifest.append({
            "fp_file": f"FP{idx}.txt",
            "doc": sec["doc"],
            "section_id": sec["section_id"],
            "section_title": sec["section_title"],
            "text_chars": len(sec["text"]),
        })

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys())
        w.writeheader()
        w.writerows(manifest)

    print(f"  {len(manifest)} FP files -> {fp_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  NOTE: No COSMIC ground truth available for this dataset.")


def main():
    print(f"Scanning XML files in {XML_DIR} ...")
    xml_files = sorted(XML_DIR.glob("*.xml"))
    print(f"  Found {len(xml_files)} XML files")

    all_sections = []
    for xf in xml_files:
        secs = extract_sections(xf)
        print(f"  {xf.name}: {len(secs)} qualifying sections")
        all_sections.extend(secs)

    print(f"\n  Total qualifying sections: {len(all_sections)}")

    OUT.mkdir(parents=True, exist_ok=True)
    generate_fp_prompts(all_sections, OUT)


if __name__ == "__main__":
    main()
