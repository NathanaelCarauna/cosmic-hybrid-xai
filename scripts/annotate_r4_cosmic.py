#!/usr/bin/env python3.10
"""annotate_r4_cosmic.py
------------------------
Usa claude-opus-4-8 para anotar 100 user stories com COSMIC CFP.

Saídas:
  datasets/user-stories-100-annotated-ai.csv   ← anotações do Opus (você revisa)
  datasets/user-stories-100-kappa-blank.csv    ← versão limpa para o 2º anotador

Uso:
  ANTHROPIC_API_KEY=sk-ant-... python3.10 scripts/annotate_r4_cosmic.py

Retoma de onde parou se interrompido (lê progress.json).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TEMPLATE = BASE / "datasets" / "user-stories-100-template.csv"
OUT_ANNOTATED = BASE / "datasets" / "user-stories-100-annotated-ai.csv"
OUT_KAPPA = BASE / "datasets" / "user-stories-100-kappa-blank.csv"
PROGRESS = BASE / "datasets" / "r4-annotation-progress.json"

MODEL = "claude-opus-4-8"

SYSTEM = """You are an expert COSMIC ISO 19761 functional size measurement analyst.
Your task: given a software user story, identify all COSMIC data movements and count CFP.

COSMIC data movement types:
- Entry (E): data enters a functional process FROM a user or external FP (input form submit, file upload, parameter received)
- Exit (X): data leaves a functional process TO a user or external FP (displayed result, report, notification sent, file downloaded)
- Read (R): data retrieved FROM persistent storage within the process (database query, file read, config lookup)
- Write (W): data written TO persistent storage (database insert/update/delete, file write, cache update)

Rules:
1. Each unique data group that moves counts as 1 data movement.
2. A typical CRUD operation = E(input) + R(existing data) + W(updated data) + X(confirmation) = 4 CFP.
3. A view/list operation = E(query params) + R(data) + X(display) = 3 CFP, or just R + X = 2 CFP.
4. Minimum CFP = 1; typical range for one user story = 2–8 CFP.
5. Focus on the WANT clause (main function). The SO THAT clause describes business goal, not extra movements.
6. Technical stories (developer wanting to "log X", "add Y to DB") → mostly W/R movements.
7. Notification stories → include X (notification sent as an Exit).

Respond ONLY with valid JSON, no markdown fences:
{
  "entries": <int>,
  "exits": <int>,
  "reads": <int>,
  "writes": <int>,
  "cfp": <int>,
  "reasoning": "<one concise sentence explaining the main movements>"
}"""

USER_TEMPLATE = """User story: {text}

Annotate with COSMIC data movements."""


def load_stories() -> list[dict]:
    with TEMPLATE.open(newline="") as f:
        return list(csv.DictReader(f))


def load_progress() -> dict[str, dict]:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {}


def save_progress(done: dict[str, dict]) -> None:
    PROGRESS.write_text(json.dumps(done, indent=2))


def annotate_one(client, story: dict) -> dict:
    """Call Opus and return annotation dict."""
    import anthropic
    msg = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(text=story["text"])}],
    )
    raw = msg.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        print("Usage: ANTHROPIC_API_KEY=sk-ant-... python3.10 scripts/annotate_r4_cosmic.py")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic not installed. Run: python3.10 -m pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    stories = load_stories()
    done = load_progress()

    print(f"Loaded {len(stories)} user stories. Already done: {len(done)}.")

    for i, story in enumerate(stories):
        us_id = story["us_id"]
        if us_id in done:
            continue

        print(f"  [{i+1:03d}/{len(stories)}] {us_id} — {story['text'][:60]}...", end=" ", flush=True)
        try:
            ann = annotate_one(client, story)
            done[us_id] = ann
            save_progress(done)
            print(f"CFP={ann['cfp']} (E={ann['entries']} X={ann['exits']} R={ann['reads']} W={ann['writes']})")
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(2)
            continue

        # Gentle rate limit: ~1 req/s
        time.sleep(1.0)

    print(f"\nDone: {len(done)}/{len(stories)} annotated.")

    # --- Write annotated CSV ---
    ann_fields = [
        "us_id", "project", "text", "kappa_subset",
        "cfp_annotator1", "cfp_annotator2",
        "entries", "exits", "reads", "writes",
        "notes",
    ]
    with OUT_ANNOTATED.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ann_fields)
        w.writeheader()
        for story in stories:
            us_id = story["us_id"]
            ann = done.get(us_id, {})
            w.writerow({
                "us_id": us_id,
                "project": story["project"],
                "text": story["text"],
                "kappa_subset": story["kappa_subset"],
                "cfp_annotator1": ann.get("cfp", ""),
                "cfp_annotator2": "",
                "entries": ann.get("entries", ""),
                "exits": ann.get("exits", ""),
                "reads": ann.get("reads", ""),
                "writes": ann.get("writes", ""),
                "notes": ann.get("reasoning", ""),
            })
    print(f"Annotated CSV → {OUT_ANNOTATED}")

    # --- Write kappa blank CSV (for 2nd annotator — kappa_subset only, no CFP) ---
    kappa_stories = [s for s in stories if s["kappa_subset"] == "True"]
    kappa_fields = ["us_id", "project", "text", "cfp_annotator", "entries", "exits", "reads", "writes", "notes"]
    with OUT_KAPPA.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kappa_fields)
        w.writeheader()
        for story in kappa_stories:
            w.writerow({
                "us_id": story["us_id"],
                "project": story["project"],
                "text": story["text"],
                "cfp_annotator": "",
                "entries": "",
                "exits": "",
                "reads": "",
                "writes": "",
                "notes": "",
            })
    print(f"Kappa blank CSV ({len(kappa_stories)} stories) → {OUT_KAPPA}")
    print("\nNext steps:")
    print("  1. Review user-stories-100-annotated-ai.csv — correct cfp_annotator1 where needed")
    print("  2. Send user-stories-100-kappa-blank.csv to 2nd annotator (they fill cfp_annotator)")
    print("  3. Run: python3.10 scripts/compute_annotation_kappa.py")


if __name__ == "__main__":
    main()
