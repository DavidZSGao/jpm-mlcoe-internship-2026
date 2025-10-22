"""Credit risk modelling utilities for Question 1 bonus tasks."""

from .altman import (
    AltmanInputs,
    AltmanResult,
    assign_rating_bucket,
    compute_altman_z,
    derive_altman_inputs,
)
from .loan_pricing import (
    DEFAULT_SPREADS,
    LoanPricingParameters,
    LoanPricingResult,
    compute_pricing,
    normalise_spread_table,
    price_scenarios,
)

__all__ = [
    "AltmanInputs",
    "AltmanResult",
    "assign_rating_bucket",
    "compute_altman_z",
    "derive_altman_inputs",
    "DEFAULT_SPREADS",
    "LoanPricingParameters",
    "LoanPricingResult",
    "compute_pricing",
    "normalise_spread_table",
    "price_scenarios",
]
