"""E8: Ensemble Normalizer — combines multiple normalizers via voting.
E9: Adaptive Normalizer — routes to NLP or SLM based on input complexity.
E11: Correction Normalizer — applies learned correction to SLM output.
E24: SmartEnsembleNormalizer — tiered correction + routing + store inventory.
E27: HierarchicalDecompositionNormalizer — chunk-based decomposition for large UCs.

Meta-normalizers that wrap other normalizers for improved accuracy.
"""
from __future__ import annotations

import re
from collections import defaultdict
from statistics import median
from typing import Optional

from ..schema import (
    BoundaryContext,
    DataMovement,
    FunctionalProcess,
    Metadata,
    MovementType,
    NormalizedRequirement,
    SourceFormat,
)
from .base import BaseNormalizer


class EnsembleNormalizer(BaseNormalizer):
    """Runs multiple normalizers and takes the median CFP prediction.

    The IR from the normalizer whose CFP count is closest to the
    median is returned (to keep valid IR structure).
    """

    def __init__(self, normalizers: list[BaseNormalizer]):
        self.normalizers = normalizers

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        results = []
        for norm in self.normalizers:
            nr = norm.normalize(text, source_format, fp_id=fp_id, metadata=metadata)
            cfp = sum(len(fp.data_movements) for fp in nr.functional_processes)
            results.append((cfp, nr))

        cfps = [r[0] for r in results]
        med = median(cfps)

        best_nr = min(results, key=lambda r: abs(r[0] - med))[1]
        return best_nr


class AdaptiveNormalizer(BaseNormalizer):
    """Routes to different normalizers based on input complexity.

    Simple inputs (sentences, short UCs) → fast normalizer (NLP/keyword)
    Complex inputs (long UCs with many steps) → SLM normalizer
    """

    def __init__(
        self,
        simple_normalizer: BaseNormalizer,
        complex_normalizer: BaseNormalizer,
        step_threshold: int = 8,
    ):
        self.simple = simple_normalizer
        self.complex = complex_normalizer
        self.step_threshold = step_threshold

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        use_complex = False

        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(text)
            n_steps = len(fields["steps"]) + len(fields["alt_flows"]) + len(fields["exceptions"])
            use_complex = n_steps >= self.step_threshold
        elif source_format == SourceFormat.SENTENCE:
            use_complex = False
        elif source_format == SourceFormat.UC_NAME:
            use_complex = False

        normalizer = self.complex if use_complex else self.simple
        return normalizer.normalize(text, source_format, fp_id=fp_id, metadata=metadata)


class CorrectionNormalizer(BaseNormalizer):
    """Wraps an SLM normalizer and applies a learned correction factor.

    Based on the observation that SLMs systematically under-count by a
    factor proportional to UC complexity. The correction formula is:

        raw = alpha * max(0, n_steps - beta)
        correction = min(raw, slm_cfp * gamma)   # cap prevents over-shoot

    where alpha, beta, gamma are learned from training data. Default
    values are calibrated on De Vito ALBERGATE (62 FPs).
    """

    def __init__(
        self,
        base_normalizer: BaseNormalizer,
        alpha: float = 0.35,
        beta: float = 8.0,
        gamma: float = 0.6,
    ):
        self.base = base_normalizer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        nr = self.base.normalize(text, source_format, fp_id=fp_id, metadata=metadata)

        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(text)
            n_steps = len(fields["steps"]) + len(fields["alt_flows"]) + len(fields["exceptions"])
            raw_correction = self.alpha * max(0, n_steps - self.beta)

            slm_cfp = sum(len(fp.data_movements) for fp in nr.functional_processes)
            max_correction = slm_cfp * self.gamma
            correction = min(raw_correction, max_correction)

            if correction > 0.5:
                n_extra = round(correction)
                for fp in nr.functional_processes:
                    for i in range(n_extra):
                        fp.data_movements.append(DataMovement(
                            type=MovementType.R,
                            data_group=f"correction_read_{i}",
                            object_of_interest=f"correction_{i}",
                        ))

        return nr


# ---------------------------------------------------------------------------
# E24a: Tiered Correction Normalizer
# ---------------------------------------------------------------------------

class TieredCorrectionNormalizer(BaseNormalizer):
    """Two-tier complexity correction: separate parameters for medium and
    large UCs.  Addresses E20's systematic under-counting on large FPs
    (GT >= 9) which contribute 55% of total MAE.

    Tier 1 (medium, steps < large_threshold): original E20 parameters.
    Tier 2 (large, steps >= large_threshold): more aggressive correction.
    """

    def __init__(
        self,
        base_normalizer: BaseNormalizer,
        *,
        alpha: float = 0.35,
        beta: float = 8.0,
        gamma: float = 0.6,
        alpha_large: float = 0.50,
        beta_large: float = 6.0,
        gamma_large: float = 0.8,
        large_threshold: int = 13,
    ):
        self.base = base_normalizer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha_large = alpha_large
        self.beta_large = beta_large
        self.gamma_large = gamma_large
        self.large_threshold = large_threshold

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        nr = self.base.normalize(text, source_format, fp_id=fp_id, metadata=metadata)

        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(text)
            n_steps = len(fields["steps"]) + len(fields["alt_flows"]) + len(fields["exceptions"])

            if n_steps >= self.large_threshold:
                a, b, g = self.alpha_large, self.beta_large, self.gamma_large
            else:
                a, b, g = self.alpha, self.beta, self.gamma

            raw_correction = a * max(0, n_steps - b)
            slm_cfp = sum(len(fp.data_movements) for fp in nr.functional_processes)
            correction = min(raw_correction, slm_cfp * g)

            if correction > 0.5:
                n_extra = round(correction)
                for fp in nr.functional_processes:
                    for i in range(n_extra):
                        fp.data_movements.append(DataMovement(
                            type=MovementType.R,
                            data_group=f"correction_read_{i}",
                            object_of_interest=f"correction_{i}",
                        ))

        return nr


# ---------------------------------------------------------------------------
# E24c: Data-store inventory verification
# ---------------------------------------------------------------------------

_STORE_INDICATORS = re.compile(
    r"\b(record|file|storage|table|store|database|persistent|"
    r"reservation|room|service|customer|invoice|booking|"
    r"account|order|product|commission|season|configuration|"
    r"age|header)\b",
    re.IGNORECASE,
)


def _extract_store_names(text: str) -> set[str]:
    """Extract candidate data-store names from raw UC text.

    Scans for noun phrases near store-indicator words in the exit/entry
    conditions and step actions.  Returns lowercased store labels.
    """
    stores: set[str] = set()
    for m in _STORE_INDICATORS.finditer(text):
        word = m.group(0).lower()
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        context = text[start:end].lower()
        if any(kw in context for kw in (
            "save", "store", "persist", "record", "delet", "remov",
            "read", "search", "retriev", "fetch", "load",
            "storage", "persistent", "file", "table", "database",
        )):
            stores.add(word)
    return stores


def _ir_store_names(nr: NormalizedRequirement) -> set[str]:
    """Collect all data_group labels from the IR (lowercased)."""
    names: set[str] = set()
    for fp in nr.functional_processes:
        for dm in fp.data_movements:
            if dm.data_group:
                for token in dm.data_group.lower().split():
                    token = token.strip("(),.")
                    if len(token) > 2:
                        names.add(token)
    return names


def verify_store_inventory(
    nr: NormalizedRequirement,
    raw_text: str,
    *,
    min_steps: int = 13,
    n_steps: int = 0,
) -> NormalizedRequirement:
    """Post-processor: add Read movements for stores mentioned in text
    but absent from the IR.  Only activates for large UCs."""
    if n_steps < min_steps:
        return nr
    text_stores = _extract_store_names(raw_text)
    ir_stores = _ir_store_names(nr)
    missing = text_stores - ir_stores
    if not missing:
        return nr

    for fp in nr.functional_processes:
        for store_name in sorted(missing):
            fp.data_movements.append(DataMovement(
                type=MovementType.R,
                data_group=f"inventory_{store_name}",
                object_of_interest=store_name,
            ))
    return nr


# ---------------------------------------------------------------------------
# E24b: Smart Ensemble Normalizer
# ---------------------------------------------------------------------------

class SmartEnsembleNormalizer(BaseNormalizer):
    """Step-count-based routing + tiered correction + store inventory.

    Routing strategy (Use Case inputs):
      - steps < medium_threshold: E20 (SLM + standard correction) only
      - medium_threshold <= steps < large_threshold: run E20 + E23,
        pick the prediction closest to their mean (complementary biases)
      - steps >= large_threshold: E20 with boosted correction + store
        inventory post-processor

    For non-Use-Case inputs, always uses the primary (E20) normalizer.
    """

    def __init__(
        self,
        primary_normalizer: BaseNormalizer,
        rewrite_normalizer: BaseNormalizer,
        *,
        alpha: float = 0.35,
        beta: float = 8.0,
        gamma: float = 0.6,
        alpha_large: float = 0.50,
        beta_large: float = 6.0,
        gamma_large: float = 0.8,
        medium_threshold: int = 6,
        large_threshold: int = 13,
    ):
        self.primary = primary_normalizer
        self.rewrite = rewrite_normalizer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha_large = alpha_large
        self.beta_large = beta_large
        self.gamma_large = gamma_large
        self.medium_threshold = medium_threshold
        self.large_threshold = large_threshold

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        if source_format != SourceFormat.USE_CASE:
            return self.primary.normalize(
                text, source_format, fp_id=fp_id, metadata=metadata,
            )

        fields = self.parse_uc_fields(text)
        n_steps = (
            len(fields["steps"])
            + len(fields["alt_flows"])
            + len(fields["exceptions"])
        )

        if n_steps < self.medium_threshold:
            nr = self.primary.normalize(
                text, source_format, fp_id=fp_id, metadata=metadata,
            )
            nr = self._apply_correction(
                nr, n_steps, self.alpha, self.beta, self.gamma,
            )
            return nr

        if n_steps >= self.large_threshold:
            nr = self.primary.normalize(
                text, source_format, fp_id=fp_id, metadata=metadata,
            )
            nr = self._apply_correction(
                nr, n_steps, self.alpha_large, self.beta_large, self.gamma_large,
            )
            nr = verify_store_inventory(nr, text, n_steps=n_steps)
            return nr

        nr_primary = self.primary.normalize(
            text, source_format, fp_id=fp_id, metadata=metadata,
        )
        nr_primary = self._apply_correction(
            nr_primary, n_steps, self.alpha, self.beta, self.gamma,
        )

        nr_rewrite = self.rewrite.normalize(
            text, source_format, fp_id=fp_id, metadata=metadata,
        )

        cfp_primary = sum(
            len(fp.data_movements) for fp in nr_primary.functional_processes
        )
        cfp_rewrite = sum(
            len(fp.data_movements) for fp in nr_rewrite.functional_processes
        )

        target = (cfp_primary + cfp_rewrite) / 2.0
        if abs(cfp_primary - target) <= abs(cfp_rewrite - target):
            return nr_primary
        return nr_rewrite

    @staticmethod
    def _apply_correction(
        nr: NormalizedRequirement,
        n_steps: int,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> NormalizedRequirement:
        raw_correction = alpha * max(0, n_steps - beta)
        slm_cfp = sum(len(fp.data_movements) for fp in nr.functional_processes)
        correction = min(raw_correction, slm_cfp * gamma)

        if correction > 0.5:
            n_extra = round(correction)
            for fp in nr.functional_processes:
                for i in range(n_extra):
                    fp.data_movements.append(DataMovement(
                        type=MovementType.R,
                        data_group=f"correction_read_{i}",
                        object_of_interest=f"correction_{i}",
                    ))
        return nr


# ---------------------------------------------------------------------------
# E26c: Self-Consistency Normalizer
# ---------------------------------------------------------------------------

class SelfConsistencyNormalizer(BaseNormalizer):
    """Generate N samples from the base normalizer, pick the one whose CFP
    is closest to the median.

    Requires a base normalizer that supports a ``temperature`` attribute
    (SLMNormalizer or OpenAINormalizer).  Temporarily overrides temperature
    to produce diverse samples.
    """

    def __init__(
        self,
        base_normalizer: BaseNormalizer,
        *,
        n_samples: int = 3,
        sample_temperature: float = 0.3,
    ):
        self.base = base_normalizer
        self.n_samples = n_samples
        self.sample_temperature = sample_temperature
        self._orig_temperature: float | None = None

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        self._orig_temperature = getattr(self.base, "temperature", None)
        if self._orig_temperature is not None:
            self.base.temperature = self.sample_temperature

        try:
            samples: list[tuple[int, NormalizedRequirement]] = []
            for _ in range(self.n_samples):
                nr = self.base.normalize(
                    text, source_format, fp_id=fp_id, metadata=metadata,
                )
                cfp = sum(len(fp.data_movements) for fp in nr.functional_processes)
                samples.append((cfp, nr))
        finally:
            if self._orig_temperature is not None:
                self.base.temperature = self._orig_temperature

        cfps = [s[0] for s in samples]
        med = median(cfps)
        best = min(samples, key=lambda s: abs(s[0] - med))
        return best[1]


# ---------------------------------------------------------------------------
# E27: Hierarchical Decomposition Normalizer
# ---------------------------------------------------------------------------

_STRIP_SUFFIXES = re.compile(
    r"\b(records?|files?|lists?|details?|display|message|"
    r"confirmation|data|info|information|new|old|previous|"
    r"available|selected|current|removal)\b",
    re.IGNORECASE,
)


def _normalise_dg(data_group: str) -> str:
    """Produce a canonical key from a data_group label for dedup."""
    dg = data_group.lower().strip()
    dg = re.sub(r"^(correction_read_\d+|inventory_\w+)$", "", dg)
    if not dg:
        return data_group.lower().strip()
    dg = _STRIP_SUFFIXES.sub("", dg)
    dg = re.sub(r"\s+", " ", dg).strip()
    dg = dg.rstrip("s") if len(dg) > 3 else dg
    return dg or data_group.lower().strip()


class HierarchicalDecompositionNormalizer(BaseNormalizer):
    """Decomposes large UCs into step-chunks for more accurate measurement.

    For UCs with many steps (>= step_threshold), the normalizer:
    1. Splits the main-scenario steps into overlapping chunks of chunk_size
    2. Builds a synthetic UC text for each chunk (same header, fewer steps)
    3. Processes each chunk with the base normalizer independently
    4. Merges all data movements with deduplication by (type, normalised_dg)

    For UCs below the threshold, delegates directly to the base normalizer.

    When ``sbert_dedup=True``, deduplication uses Sentence-BERT cosine
    similarity instead of brittle string normalisation — this handles
    cases where chunks describe the same data group with different
    surface forms (e.g. "room availability" vs "available rooms").
    """

    _sbert_model = None

    def __init__(
        self,
        base_normalizer: BaseNormalizer,
        *,
        step_threshold: int = 12,
        chunk_size: int = 7,
        chunk_overlap: int = 1,
        sbert_dedup: bool = False,
        sbert_sim_threshold: float = 0.75,
    ):
        self.base = base_normalizer
        self.step_threshold = step_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sbert_dedup = sbert_dedup
        self.sbert_sim_threshold = sbert_sim_threshold

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        if source_format != SourceFormat.USE_CASE:
            return self.base.normalize(
                text, source_format, fp_id=fp_id, metadata=metadata,
            )

        fields = self.parse_uc_fields(text)
        n_steps = (
            len(fields["steps"])
            + len(fields["alt_flows"])
            + len(fields["exceptions"])
        )

        if n_steps < self.step_threshold:
            return self.base.normalize(
                text, source_format, fp_id=fp_id, metadata=metadata,
            )

        chunks = self._make_chunks(fields["steps"])

        all_movements: list[DataMovement] = []
        chunk_nrs: list[NormalizedRequirement] = []

        for cidx, chunk_steps in enumerate(chunks):
            exc = fields["exceptions"] if cidx == len(chunks) - 1 else []
            alt = fields["alt_flows"] if cidx == len(chunks) - 1 else []
            mini_text = self._build_chunk_text(fields, chunk_steps, alt, exc)
            nr = self.base.normalize(
                mini_text, source_format, fp_id=fp_id, metadata=metadata,
            )
            chunk_nrs.append(nr)
            for fp in nr.functional_processes:
                all_movements.extend(fp.data_movements)

        if self.sbert_dedup:
            merged = self._merge_movements_sbert(
                all_movements, self.sbert_sim_threshold,
            )
        else:
            merged = self._merge_movements(all_movements)

        primary_actor = fields.get("primary_actor") or "User"
        fp = FunctionalProcess(
            name=fields.get("name") or "Unknown",
            triggering_event=fields.get("name") or "Unknown",
            primary_functional_user=primary_actor,
            data_movements=merged,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=source_format,
            raw_text=text,
            boundary=BoundaryContext(
                software_name=(
                    metadata.project if metadata and metadata.project else "Unknown"
                ),
                functional_users=[primary_actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(source_file=fp_id),
        )

    def _make_chunks(
        self, steps: list[tuple[str, str, str]],
    ) -> list[list[tuple[str, str, str]]]:
        """Split steps into overlapping chunks."""
        if len(steps) <= self.chunk_size:
            return [steps]
        chunks: list[list[tuple[str, str, str]]] = []
        stride = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(steps), stride):
            chunk = steps[start : start + self.chunk_size]
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_size >= len(steps):
                break
        return chunks

    @staticmethod
    def _build_chunk_text(
        fields: dict,
        steps: list[tuple[str, str, str]],
        alt_flows: list[tuple[str, str, str]],
        exceptions: list[tuple[str, str, str]],
    ) -> str:
        """Reconstruct a valid COSMET-format UC text from fields + step subset."""
        lines: list[str] = ["ID"]
        lines.append(f"{fields.get('id', '')}\t{fields.get('name', '')}")
        if fields.get("description"):
            lines.append(f"Description\t{fields['description']}")
        if fields.get("primary_actor"):
            lines.append(f"Primary actor\t{fields['primary_actor']}")
        if fields.get("supporting_actors"):
            lines.append(f"Supporting actors\t{fields['supporting_actors']}")
        lines.append("MAIN SCENARIO")
        for i, (_, actor, action) in enumerate(steps, 1):
            lines.append(f"{i}\t{actor}\t{action}")
        if alt_flows:
            lines.append("Alternative flow no. 1")
            for sid, actor, action in alt_flows:
                lines.append(f"{sid}\t{actor}\t{action}")
        if exceptions:
            lines.append("Exceptions")
            for sid, actor, action in exceptions:
                lines.append(f"{sid}\t{actor}\t{action}")
        return "\n".join(lines)

    @staticmethod
    def _merge_movements(
        movements: list[DataMovement],
    ) -> list[DataMovement]:
        """Deduplicate movements by (type, normalised data_group).

        Keeps exactly one triggering entry and one error/confirmation exit.
        """
        seen: dict[tuple[str, str], DataMovement] = {}
        has_trigger = False
        has_err_exit = False

        for dm in movements:
            key = (dm.type.value, _normalise_dg(dm.data_group))

            if dm.is_triggering_entry:
                if has_trigger:
                    continue
                has_trigger = True
                seen[key] = dm
                continue

            if dm.is_error_confirmation:
                if has_err_exit:
                    continue
                has_err_exit = True
                seen[key] = dm
                continue

            if dm.data_group.startswith("correction_"):
                continue

            if key not in seen:
                seen[key] = dm

        result = list(seen.values())

        if not has_trigger and result:
            for dm in result:
                if dm.type == MovementType.E:
                    dm.is_triggering_entry = True
                    break

        return result

    @classmethod
    def _get_sbert_model(cls):
        """Lazy-load Sentence-BERT model (singleton)."""
        if cls._sbert_model is None:
            from sentence_transformers import SentenceTransformer
            cls._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._sbert_model

    @classmethod
    def _merge_movements_sbert(
        cls,
        movements: list[DataMovement],
        sim_threshold: float = 0.75,
    ) -> list[DataMovement]:
        """Deduplicate movements using SBERT cosine similarity on data_group names.

        Two movements with the same type are considered duplicates if their
        data_group labels have cosine similarity >= sim_threshold.
        """
        import numpy as np

        if not movements:
            return []

        has_trigger = False
        has_err_exit = False
        kept: list[DataMovement] = []

        corrections = [dm for dm in movements if dm.data_group.startswith("correction_")]
        regular = [dm for dm in movements if not dm.data_group.startswith("correction_")]

        if not regular:
            return []

        model = cls._get_sbert_model()
        dg_texts = [dm.data_group for dm in regular]
        embeddings = model.encode(dg_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        assigned_cluster: list[int] = [-1] * len(regular)
        cluster_id = 0

        for i, dm in enumerate(regular):
            if dm.is_triggering_entry:
                if has_trigger:
                    continue
                has_trigger = True
                assigned_cluster[i] = cluster_id
                cluster_id += 1
                kept.append(dm)
                continue

            if dm.is_error_confirmation:
                if has_err_exit:
                    continue
                has_err_exit = True
                assigned_cluster[i] = cluster_id
                cluster_id += 1
                kept.append(dm)
                continue

            is_dup = False
            for j in range(len(kept)):
                if kept[j].type != dm.type:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[_kept_to_reg_idx(kept, j, regular)]))
                if sim >= sim_threshold:
                    is_dup = True
                    break

            if not is_dup:
                assigned_cluster[i] = cluster_id
                cluster_id += 1
                kept.append(dm)

        if not has_trigger and kept:
            for dm in kept:
                if dm.type == MovementType.E:
                    dm.is_triggering_entry = True
                    break

        return kept


def _kept_to_reg_idx(kept: list, kept_idx: int, regular: list) -> int:
    """Map a kept-list index back to the regular-list index."""
    target = kept[kept_idx]
    for i, dm in enumerate(regular):
        if dm is target:
            return i
    return 0


# ---------------------------------------------------------------------------
# E32: Split Normalizer — M0 atomic split before M1
# ---------------------------------------------------------------------------

class SplitNormalizer(BaseNormalizer):
    """Chains M0 (atomic sentence split) → base SLM normalizer.

    The split preprocessor rewrites compound UC steps into atomic
    single-action lines. The base normalizer runs on the split text.
    TieredCorrection is applied using the ORIGINAL step count to avoid
    inflation from the extra split sentences.
    """

    def __init__(
        self,
        split_preprocessor,
        base_normalizer: BaseNormalizer,
        *,
        apply_correction: bool = True,
        alpha: float = 0.35,
        beta: float = 8.0,
        gamma: float = 0.6,
        alpha_large: float = 0.50,
        beta_large: float = 6.0,
        gamma_large: float = 0.8,
        large_threshold: int = 13,
    ):
        from .split_preprocessor import SplitPreprocessor
        self._splitter: SplitPreprocessor = split_preprocessor
        self._base = base_normalizer
        self._apply_correction = apply_correction
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._alpha_large = alpha_large
        self._beta_large = beta_large
        self._gamma_large = gamma_large
        self._large_threshold = large_threshold

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        original_n_steps = 0
        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(text)
            original_n_steps = (
                len(fields["steps"]) + len(fields["alt_flows"])
                + len(fields["exceptions"])
            )

        split_text = self._splitter.split(text, source_format)
        nr = self._base.normalize(
            split_text, source_format, fp_id=fp_id, metadata=metadata,
        )

        if self._apply_correction and source_format == SourceFormat.USE_CASE:
            n = original_n_steps
            if n >= self._large_threshold:
                a, b, g = self._alpha_large, self._beta_large, self._gamma_large
            else:
                a, b, g = self._alpha, self._beta, self._gamma
            raw_correction = a * max(0, n - b)
            slm_cfp = sum(len(fp.data_movements) for fp in nr.functional_processes)
            correction = min(raw_correction, slm_cfp * g)
            if correction > 0.5:
                n_extra = round(correction)
                for fp in nr.functional_processes:
                    for i in range(n_extra):
                        fp.data_movements.append(DataMovement(
                            type=MovementType.R,
                            data_group=f"correction_read_{i}",
                            object_of_interest=f"correction_{i}",
                        ))

        return nr
