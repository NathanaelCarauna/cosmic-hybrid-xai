"""S4-T4 — Testes de shared/metrics.py (edge cases + regressão da errata SA)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from shared.metrics import standardized_accuracy, nmae  # noqa: E402


class TestStandardizedAccuracy:
    def test_perfect_prediction(self):
        y = [2, 4, 6, 8]
        assert standardized_accuracy(y, y) == pytest.approx(1.0)

    def test_mean_predictor_is_zero(self):
        y = np.array([2.0, 4.0, 6.0, 8.0])
        pred = np.full(4, y.mean())
        assert standardized_accuracy(y, pred) == pytest.approx(0.0)

    def test_worse_than_mean_is_negative(self):
        y = [3, 3, 4, 4]
        pred = [10, 10, 10, 10]
        assert standardized_accuracy(y, pred) < 0

    def test_constant_gt_returns_nan(self):
        # distribuição constante → MAE_baseline = 0 → SA indefinido
        y = [3, 3, 3]
        assert np.isnan(standardized_accuracy(y, [3, 3, 3]))

    def test_small_n(self):
        y = [1, 5]
        pred = [1, 5]
        assert standardized_accuracy(y, pred) == pytest.approx(1.0)

    def test_errata_sa_is_not_one_minus_nmae(self):
        """Regressão da errata 2026-06-10 (auditoria §19.11): em distribuição
        estreita, 1−NMAE infla muito o valor; SA correto usa o desvio médio.
        Cenário do colapso E33d/US: GT ~3.6, pred ~1.0."""
        y = np.array([3.0] * 55 + [4.0] * 37 + [5.0, 5.0, 6.0, 6.0, 11.0, 11.0])
        pred = np.ones_like(y)
        sa = standardized_accuracy(y, pred)
        fake = 1 - nmae(y, pred)  # a fórmula errada
        assert sa < 0, "predição constante=1 contra GT~3.6 deve ter SA negativo"
        assert fake > 0, "1−NMAE mascara o colapso (motivo da errata)"
        assert abs(sa - fake) > 1.0

    def test_narrow_distribution_sensitivity(self):
        # erro de 1 CFP em distribuição estreita: pior que a média → SA<0
        y = np.array([3, 4, 3, 4, 3, 4], dtype=float)
        pred = y + 1
        assert standardized_accuracy(y, pred) < 0


class TestNmae:
    def test_zero_error(self):
        assert nmae([2, 4], [2, 4]) == pytest.approx(0.0)

    def test_scale(self):
        # MAE=1 sobre média=4 → NMAE=0.25
        assert nmae([4, 4], [5, 3]) == pytest.approx(0.25)
