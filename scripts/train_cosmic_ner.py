#!/usr/bin/env python3
"""Train a spaCy custom NER model on COSMIC entity annotations.

Reads JSONL annotations from datasets/ner-annotations-manual/ and trains a
spaCy NER pipeline to recognize COSMIC entities: FUNC_USER, DATA_OBJ,
ACTION_VERB, UI_ELEMENT, PERSIST_STORE, CONDITION.

Anti-overfitting: system-level train/val/test split — entire systems are held
out so the model never trains on text it evaluates against.

Usage:
  python3 scripts/train_cosmic_ner.py [--epochs 50] [--patience 8]
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

BASE = Path(__file__).resolve().parent.parent
MANUAL_DIR = BASE / "datasets" / "ner-annotations-manual"
MODEL_DIR = BASE / "models" / "cosmic-ner"

LABELS = ["FUNC_USER", "DATA_OBJ", "ACTION_VERB", "UI_ELEMENT", "PERSIST_STORE", "CONDITION"]

TRAIN_SYSTEMS = ["ALBERGATE", "K01726", "K01719"]
VAL_SYSTEMS = ["AutomaticLineSwitching", "K01720"]
TEST_SYSTEMS = ["RiceCooker", "U-CURE"]


def load_from_systems(systems: list[str], ann_dir: Path) -> list[tuple[str, dict]]:
    """Load training examples from specific systems only."""
    data = []
    for system in systems:
        path = ann_dir / f"{system}.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            record = json.loads(line)
            for step in record["steps"]:
                text = step["action_text"]
                if not text:
                    continue

                entities = []
                seen_spans = set()
                for ent in step["entities"]:
                    start, end, label = ent["start"], ent["end"], ent["label"]
                    if start < 0 or end > len(text) or start >= end:
                        continue
                    if label not in LABELS:
                        continue
                    overlaps = any((s < end and e > start) for s, e in seen_spans)
                    if overlaps:
                        continue
                    seen_spans.add((start, end))
                    entities.append((start, end, label))

                if entities:
                    data.append((text, {"entities": entities}))
    return data


def evaluate(nlp, eval_data: list[tuple[str, dict]]) -> dict:
    """Evaluate NER model — span-level P/R/F1 per label + overall."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for text, annotations in eval_data:
        doc = nlp(text)
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}
        gold_ents = {(s, e, l) for s, e, l in annotations["entities"]}

        for ent in pred_ents:
            if ent in gold_ents:
                tp[ent[2]] += 1
            else:
                fp[ent[2]] += 1
        for ent in gold_ents:
            if ent not in pred_ents:
                fn[ent[2]] += 1

    results = {}
    for label in LABELS:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] > 0 else 0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        results[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    p_all = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    r_all = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if p_all + r_all > 0 else 0
    results["OVERALL"] = {"precision": round(p_all, 4), "recall": round(r_all, 4), "f1": round(f1_all, 4)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without val F1 improvement)")
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ann-dir", type=str, default=str(MANUAL_DIR),
                        help="Annotations directory")
    args = parser.parse_args()

    random.seed(args.seed)
    ann_dir = Path(args.ann_dir)

    print("System-level split (anti-overfitting):")
    print(f"  Train:  {TRAIN_SYSTEMS}")
    print(f"  Val:    {VAL_SYSTEMS}")
    print(f"  Test:   {TEST_SYSTEMS}")

    train_data = load_from_systems(TRAIN_SYSTEMS, ann_dir)
    val_data = load_from_systems(VAL_SYSTEMS, ann_dir)
    test_data = load_from_systems(TEST_SYSTEMS, ann_dir)
    print(f"\nExamples — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if not train_data:
        print("ERROR: no training data found. Aborting.")
        return

    label_counts = defaultdict(int)
    for _, annots in train_data:
        for _, _, label in annots["entities"]:
            label_counts[label] += 1
    print("Label distribution (train):")
    for label in LABELS:
        print(f"  {label:15s}: {label_counts[label]}")

    print("\nCreating spaCy NER pipeline...")
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in LABELS:
        ner.add_label(label)

    optimizer = nlp.begin_training()

    print(f"\nTraining (max {args.epochs} epochs, patience={args.patience}, dropout={args.dropout})...")
    best_f1 = 0.0
    best_epoch = 0
    stale = 0

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

        for batch in batches:
            examples = []
            for text, annots in batch:
                doc = nlp.make_doc(text)
                try:
                    examples.append(Example.from_dict(doc, annots))
                except Exception:
                    continue
            if examples:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    nlp.update(examples, drop=args.dropout, losses=losses, sgd=optimizer)

        val_results = evaluate(nlp, val_data)
        val_f1 = val_results["OVERALL"]["f1"]

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_epoch = epoch + 1
            stale = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            nlp.to_disk(MODEL_DIR / "best")
        else:
            stale += 1

        marker = " *" if improved else ""
        if (epoch + 1) % 3 == 0 or epoch == 0 or improved or stale >= args.patience:
            print(f"  Epoch {epoch+1:3d}: loss={losses.get('ner', 0):.3f}, "
                  f"val_F1={val_f1:.3f} (best={best_f1:.3f} @ {best_epoch}){marker}")

        if stale >= args.patience:
            print(f"  Early stopping — no improvement for {args.patience} epochs.")
            break

    print(f"\nBest model: {MODEL_DIR / 'best'} (val_F1={best_f1:.3f} @ epoch {best_epoch})")

    print("\n--- Val set (used for early stopping) ---")
    best_nlp = spacy.load(MODEL_DIR / "best")
    val_final = evaluate(best_nlp, val_data)
    for label, m in val_final.items():
        print(f"  {label:15s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    if test_data:
        print("\n--- Test set (held-out, never seen during training) ---")
        test_results = evaluate(best_nlp, test_data)
        for label, m in test_results.items():
            print(f"  {label:15s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    else:
        test_results = {}

    metrics = {"val": val_final, "test": test_results,
               "split": {"train": TRAIN_SYSTEMS, "val": VAL_SYSTEMS, "test": TEST_SYSTEMS},
               "best_epoch": best_epoch, "max_epochs": args.epochs, "patience": args.patience,
               "dropout": args.dropout, "seed": args.seed}
    metrics_path = MODEL_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
