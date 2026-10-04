"""Confidence intervals that widen as evidence thins.

Spread comes from re-running the geometry on interleaved frame subsets of the
same capture (sensor + estimator variability), combined in quadrature with a
fixed sensor floor, then widened for walls with little point evidence.
PROVISIONAL: constants are calibrated against ground truth in the benchmark
phase; until then treat intervals as uncalibrated.
"""
import numpy as np

K_SIGMA = 2.0           # ~95%
LEN_FLOOR_ABS = 0.01    # m
LEN_FLOOR_REL = 0.005
HEIGHT_FLOOR = 0.01
OPENING_FLOOR = 0.02
LOW_EVIDENCE_PENALTY = 0.15  # fraction of wall length at zero evidence


def ci(value, half):
    return {"value": None if value is None else float(value),
            "ci95": None if value is None else [float(value - half), float(value + half)],
            "half_width": None if half is None else float(half)}


def length_half(value, rel_spread, evidence):
    base = np.sqrt((K_SIGMA * rel_spread * value) ** 2 + (LEN_FLOOR_ABS + LEN_FLOOR_REL * value) ** 2)
    return float(base + LOW_EVIDENCE_PENALTY * value * (1.0 - evidence))


def height_half(std):
    return float(np.sqrt((K_SIGMA * std) ** 2 + HEIGHT_FLOOR ** 2))


def area_half(value, rel_spread):
    return float(np.sqrt((K_SIGMA * rel_spread * value) ** 2 + (2 * LEN_FLOOR_REL * value) ** 2))


def opening_half(width):
    return float(OPENING_FLOOR + 0.01 * width)


def widen(half, value, prior_rel):
    """Add an a-priori relative error (sensor tier prior) in quadrature."""
    if value is None or prior_rel <= 0:
        return half
    return float(np.sqrt(half ** 2 + (prior_rel * value) ** 2))
