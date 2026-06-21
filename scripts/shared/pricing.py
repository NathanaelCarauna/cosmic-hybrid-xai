"""Model pricing table and cost estimation utilities.

Prices are per 1M tokens (USD) as of early 2026. Update as needed.
"""
from __future__ import annotations

PRICING: dict[str, dict[str, float]] = {
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
}

LOCAL_ZERO = {"input": 0.00, "output": 0.00}

_FALLBACK = {"input": 10.00, "output": 30.00}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """Return estimated cost in USD for a given token count and model."""
    if model.startswith("local:") or model.startswith("local/"):
        return 0.0
    prices = PRICING.get(model, _FALLBACK)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def get_model_prices(model: str) -> dict[str, float]:
    """Return the price dict for a model, falling back to gpt-4-turbo rates."""
    return PRICING.get(model, _FALLBACK)
