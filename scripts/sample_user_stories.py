#!/usr/bin/env python3
"""sample_user_stories.py
--------------------------
Sample 100 User Stories from the Dalpiaz Mendeley Data dataset for COSMIC CFP
annotation (R4 track, Sprint 2).

Sampling strategy:
  - Stratified by project: ~5 US per project across 22 projects ≈ 100 total
  - Filter out stories with very short action clauses (< 5 words)
  - If fewer than 5 stories per project, take all of them
  - Final sample trimmed/padded to exactly 100 (or as close as possible)
  - Sub-sample of 20 for Cohen's κ (seed=42, flagged with kappa_subset=True)

Output files:
  datasets/cross-sample/r4-100-us.csv
  datasets/cross-sample/r4-100-us-manifest.json
  datasets/user-stories-100-template.csv

Raw dataset location:
  datasets/raw/user-stories-dalpiaz/

If the raw dataset is not found, prints download instructions and exits.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent  # software-size-estimator/

RAW_DIR = BASE / "datasets" / "raw" / "user-stories-dalpiaz"
OUT_DIR = BASE / "datasets" / "cross-sample"
TEMPLATE_OUT = BASE / "datasets" / "user-stories-100-template.csv"

SEED = 42
TARGET_TOTAL = 100
TARGET_PER_PROJECT = 5
KAPPA_N = 20
MIN_ACTION_WORDS = 5

# ---------------------------------------------------------------------------
# Column name aliases (the Dalpiaz dataset has been released in variants)
# ---------------------------------------------------------------------------
ROLE_COLS = ["role", "actor", "Role", "Actor"]
ACTION_COLS = ["action", "means", "Action", "Means"]
BENEFIT_COLS = ["benefit", "ends", "Benefit", "Ends", "goal", "Goal"]
PROJECT_COLS = ["project_id", "project", "Project", "ProjectID", "project-id"]
ID_COLS = ["story_id", "id", "ID", "StoryID", "story-id"]

MENDELEY_URL = "https://data.mendeley.com/datasets/7zbk8zsd8y/1"
MENDELEY_INSTRUCTIONS = f"""
Dataset not found: {RAW_DIR}

To download the Dalpiaz Mendeley Data dataset:
  1. Visit: {MENDELEY_URL}
  2. Click "Download All" (or download the individual CSV file).
  3. Extract the contents into:
         {RAW_DIR}/
  4. The directory should contain a CSV file (commonly named 'user_stories.csv'
     or similar) with columns for story_id/id, role/actor, action/means,
     benefit/ends, and project_id/project.
  5. Re-run this script.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(header: list[str], candidates: list[str]) -> str | None:
    """Return the first matching column name from candidates, or None."""
    for c in candidates:
        if c in header:
            return c
    return None


def _word_count(text: str) -> int:
    """Count whitespace-separated tokens in a string."""
    return len(text.split())


def _assemble_us_text(role: str, action: str, benefit: str) -> str:
    """Build the canonical Connextra-format US string."""
    role = role.strip()
    action = action.strip()
    benefit = benefit.strip()
    if benefit:
        return f"As a {role}, I want {action} so that {benefit}."
    return f"As a {role}, I want {action}."


def _find_raw_csv(raw_dir: Path) -> Path | None:
    """Search for the main CSV file in the raw dataset directory."""
    candidates = [
        "user_stories.csv",
        "user-stories.csv",
        "stories.csv",
        "data.csv",
        "dataset.csv",
        "g02-uc.csv",  # known Dalpiaz filename variant
    ]
    # Try known names first
    for name in candidates:
        p = raw_dir / name
        if p.exists():
            return p
    # Fall back: any CSV in root of raw_dir
    csvs = list(raw_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    # Recurse one level (zip may have created a sub-folder)
    for sub in raw_dir.iterdir():
        if sub.is_dir():
            for name in candidates:
                p = sub / name
                if p.exists():
                    return p
            sub_csvs = list(sub.glob("*.csv"))
            if sub_csvs:
                return sub_csvs[0]
    return None


def load_dataset(raw_dir: Path) -> tuple[list[dict], str, str, str, str, str]:
    """Load the raw Dalpiaz CSV and return rows + resolved column names."""
    if not raw_dir.exists():
        print(MENDELEY_INSTRUCTIONS, file=sys.stderr)
        sys.exit(1)

    csv_path = _find_raw_csv(raw_dir)
    if csv_path is None:
        print(
            f"[ERROR] No CSV file found inside {raw_dir}.\n{MENDELEY_INSTRUCTIONS}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[info] Loading dataset from: {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []

    print(f"[info] Loaded {len(rows)} rows, {len(header)} columns: {header}")

    id_col = _find_col(header, ID_COLS)
    role_col = _find_col(header, ROLE_COLS)
    action_col = _find_col(header, ACTION_COLS)
    benefit_col = _find_col(header, BENEFIT_COLS)
    project_col = _find_col(header, PROJECT_COLS)

    missing = [
        name
        for name, col in [
            ("role/actor", role_col),
            ("action/means", action_col),
            ("project_id/project", project_col),
        ]
        if col is None
    ]
    if missing:
        print(
            f"[ERROR] Could not find required columns: {missing}\n"
            f"Available columns: {header}\n"
            "Please verify the dataset format matches the Dalpiaz Mendeley Data schema.",
            file=sys.stderr,
        )
        sys.exit(1)

    return rows, id_col, role_col, action_col, benefit_col, project_col


def filter_rows(
    rows: list[dict],
    role_col: str,
    action_col: str,
    benefit_col: str | None,
    project_col: str,
    id_col: str | None,
) -> list[dict]:
    """Filter out rows with missing required fields or too-short action clauses."""
    kept = []
    skipped_empty = 0
    skipped_short = 0
    for i, row in enumerate(rows):
        role = row.get(role_col, "").strip()
        action = row.get(action_col, "").strip()
        project = row.get(project_col, "").strip()

        if not role or not action or not project:
            skipped_empty += 1
            continue
        if _word_count(action) < MIN_ACTION_WORDS:
            skipped_short += 1
            continue
        kept.append(row)

    print(
        f"[info] Filtering: {len(rows)} total → {len(kept)} kept "
        f"(skipped {skipped_empty} empty, {skipped_short} short-action)"
    )
    return kept


def stratified_sample(
    rows: list[dict],
    project_col: str,
    target_per_project: int,
    target_total: int,
    rng: random.Random,
) -> list[dict]:
    """
    Sample target_per_project stories per project.
    If a project has fewer, take all of them.
    Trim or top-up to exactly target_total if needed.
    """
    # Group by project
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        proj = row[project_col].strip()
        buckets.setdefault(proj, []).append(row)

    projects = sorted(buckets.keys())
    print(f"[info] Projects found: {len(projects)}")
    for p in projects:
        print(f"       {p}: {len(buckets[p])} eligible stories")

    sample: list[dict] = []
    for proj in projects:
        pool = buckets[proj]
        n = min(target_per_project, len(pool))
        picked = rng.sample(pool, n)
        sample.extend(picked)

    print(f"[info] Stratified sample before trim: {len(sample)} stories")

    if len(sample) > target_total:
        # Trim: remove from projects with largest representation first
        # (maintain fairness: prefer uniform distribution)
        sample_shuffled = list(sample)
        rng.shuffle(sample_shuffled)
        sample = sample_shuffled[:target_total]
        print(f"[info] Trimmed to {target_total}")
    elif len(sample) < target_total:
        # Top-up from the remaining pool (not yet selected)
        selected_ids = {id(r) for r in sample}
        remaining = [r for r in rows if id(r) not in selected_ids]
        rng.shuffle(remaining)
        needed = target_total - len(sample)
        topup = remaining[:needed]
        sample.extend(topup)
        print(
            f"[info] Top-up: added {len(topup)} from overflow pool → "
            f"total {len(sample)}"
        )

    return sample


def build_output_rows(
    sample: list[dict],
    id_col: str | None,
    role_col: str,
    action_col: str,
    benefit_col: str | None,
    project_col: str,
    kappa_indices: set[int],
) -> list[dict]:
    """Build output rows in the canonical output schema."""
    out = []
    for i, row in enumerate(sample):
        role = row.get(role_col, "").strip()
        action = row.get(action_col, "").strip()
        benefit = row.get(benefit_col, "").strip() if benefit_col else ""
        project = row.get(project_col, "").strip()

        # Build us_id
        raw_id = row.get(id_col, "") if id_col else ""
        raw_id = str(raw_id).strip()
        us_id = f"dalpiaz-{raw_id}" if raw_id else f"dalpiaz-{i+1:03d}"

        us_text = _assemble_us_text(role, action, benefit)

        out.append({
            "us_id": us_id,
            "project_id": project,
            "us_text": us_text,
            "us_text_role": role,
            "us_text_action": action,
            "us_text_benefit": benefit,
            "kappa_subset": "true" if i in kappa_indices else "false",
        })
    return out


def write_sample_csv(out_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "us_id", "project_id", "us_text",
        "us_text_role", "us_text_action", "us_text_benefit",
        "kappa_subset",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[out] Sample CSV → {path}  ({len(out_rows)} rows)")


def write_manifest(
    out_rows: list[dict],
    path: Path,
    n_original: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    projects = sorted({r["project_id"] for r in out_rows})
    dist: dict[str, int] = {}
    for r in out_rows:
        dist[r["project_id"]] = dist.get(r["project_id"], 0) + 1

    manifest = {
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_original_dataset": n_original,
        "n_eligible_after_filter": n_original,  # overwritten below
        "n_sample": len(out_rows),
        "n_kappa": KAPPA_N,
        "min_action_words_filter": MIN_ACTION_WORDS,
        "target_per_project": TARGET_PER_PROJECT,
        "projects": projects,
        "n_projects": len(projects),
        "distribution_by_project": dist,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[out] Manifest JSON → {path}")
    return manifest


def write_template_csv(out_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "us_id", "project_id", "us_text",
        "us_text_role", "us_text_action", "us_text_benefit",
        "kappa_subset",
    ]
    annotation_fields = [
        "count_entry",
        "count_exit",
        "count_read",
        "count_write",
        "cfp_total",
        "movements_detail",
        "ambiguity_flag",
        "notes",
        "annotator_id",
        "annotated_at",
    ]
    fieldnames = base_fields + annotation_fields

    # Build template rows: base data + empty annotation columns
    template_rows = []
    for row in out_rows:
        tr = {k: row.get(k, "") for k in base_fields}
        for af in annotation_fields:
            tr[af] = ""
        template_rows.append(tr)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(template_rows)
    print(f"[out] Annotator template → {path}  ({len(template_rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)

    # 1. Load dataset
    rows, id_col, role_col, action_col, benefit_col, project_col = load_dataset(RAW_DIR)
    n_original = len(rows)

    # 2. Filter
    eligible = filter_rows(rows, role_col, action_col, benefit_col, project_col, id_col)
    n_eligible = len(eligible)

    if n_eligible == 0:
        print(
            "[ERROR] No eligible stories after filtering. "
            "Check dataset format and column mapping.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Stratified sample
    sample = stratified_sample(
        eligible, project_col, TARGET_PER_PROJECT, TARGET_TOTAL, rng
    )

    # 4. Select κ subset (20 random from the 100)
    kappa_indices: set[int] = set(
        random.Random(SEED).sample(range(len(sample)), min(KAPPA_N, len(sample)))
    )

    # 5. Build output rows
    out_rows = build_output_rows(
        sample, id_col, role_col, action_col, benefit_col, project_col, kappa_indices
    )

    # 6. Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_csv = OUT_DIR / "r4-100-us.csv"
    manifest_json = OUT_DIR / "r4-100-us-manifest.json"

    write_sample_csv(out_rows, sample_csv)

    manifest = write_manifest(out_rows, manifest_json, n_original)
    # patch n_eligible into manifest after the fact
    manifest["n_eligible_after_filter"] = n_eligible
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    write_template_csv(out_rows, TEMPLATE_OUT)

    # 7. Summary
    kappa_count = sum(1 for r in out_rows if r["kappa_subset"] == "true")
    print()
    print("=" * 60)
    print("Sample complete")
    print(f"  Total stories in dataset : {n_original}")
    print(f"  Eligible after filter    : {n_eligible}")
    print(f"  Sample size              : {len(out_rows)}")
    print(f"  κ subset                 : {kappa_count}")
    print(f"  Projects represented     : {manifest['n_projects']}")
    print()
    print("Output files:")
    print(f"  {sample_csv}")
    print(f"  {manifest_json}")
    print(f"  {TEMPLATE_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
