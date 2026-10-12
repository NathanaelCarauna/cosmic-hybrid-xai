"""E43 — Format-aware few-shot selection for the OpenAI normalizer.

Motivation (see docs/estimativa-cosmic-tese/resultados-e40-llm-normalizador.md):
E40 showed that the shared few-shot set `fewshots/v2-clean/` has unbalanced format
coverage (5/6 use case, 1/6 sentence, ZERO for UC names and Connextra user stories),
and that the two zero-coverage cells were the two worst. E42 then showed that a
stronger model (gpt-4o) closes roughly half the gap but not all of it.

This normalizer selects the few-shot block by `source_format` at call time, drawing
from `fewshots/v3-format/<bucket>/`. Everything else (prompts, schema, parsing,
downstream 26-rule engine) is unchanged, so the comparison against E40/E42 is clean.

Provenance of the new examples: transliterated from EXPERT-ANNOTATED held-out data,
never from evaluated items.
  - connextra/ : dev-20 (`kappa_subset == True`) of
                 datasets/user-stories-100-annotated-cosmic-expert.csv, which carries
                 the per-movement breakdown (entries/exits/reads/writes) and the
                 annotator's reasoning. Evaluation must therefore run on the clean
                 test-78, NOT on the full 98.
  - sentence/  : datasets/holdout/zenodo_train.csv (zero overlap with sample-300).
Entities and domains were transliterated; movement patterns and counts were preserved
verbatim from the expert annotation.

UC names (Ochodek) are deliberately NOT covered: that dataset's `TransTypes` is
CRUD-typed (C|D|R|U), not COSMIC E/X/R/W, and its unit differs across files, so
building examples there would require inference rather than transliteration.
"""
from __future__ import annotations

from pathlib import Path

from .openai_normalizer import OpenAINormalizer

_V3_DIR = Path(__file__).parent / "fewshots" / "v3-format"

# SourceFormat value -> bucket directory under v3-format/
_BUCKETS = {
    "use_case": "use_case",
    "sentence": "sentence",
    "user_story": "connextra",
    "connextra": "connextra",
}
_DEFAULT_BUCKET = "use_case"


def _load_bucket(bucket: str) -> str:
    d = _V3_DIR / bucket
    examples = sorted(d.glob("example_*.txt"))
    if not examples:
        raise FileNotFoundError(f"No few-shot examples in {d}")
    return "\n\n".join(p.read_text(encoding="utf-8").strip() for p in examples)


class FormatAwareOpenAINormalizer(OpenAINormalizer):
    """OpenAINormalizer that swaps the few-shot block per source_format."""

    def __init__(self, *args, force_bucket: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # The runner only exposes --source-format {use_case, sentence, uc_name}, so
        # Connextra user stories arrive tagged as "sentence". force_bucket overrides
        # the mapping for those runs.
        if force_bucket and not (_V3_DIR / force_bucket).is_dir():
            raise ValueError(
                f"Unknown few-shot bucket {force_bucket!r}; "
                f"available: {sorted(p.name for p in _V3_DIR.iterdir() if p.is_dir())}"
            )
        self._force_bucket = force_bucket
        self._cache: dict[str, str] = {}

    def _bucket_for(self, source_format) -> str:
        if self._force_bucket:
            return self._force_bucket
        raw = getattr(source_format, "value", source_format)
        return _BUCKETS.get(str(raw), _DEFAULT_BUCKET)

    def normalize(self, text, source_format, **kwargs):
        bucket = self._bucket_for(source_format)
        if bucket not in self._cache:
            self._cache[bucket] = _load_bucket(bucket)
        # OpenAINormalizer.normalize() reads self._fewshot_text when building messages.
        self._fewshot_text = self._cache[bucket]
        return super().normalize(text, source_format, **kwargs)
