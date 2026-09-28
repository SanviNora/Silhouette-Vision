"""Calibrated "is the best result the exact product?" (scripts/06_match_confidence.py).

Raw similarity is not a probability and alone never exceeded ~64% precision on LookBench. What
separates an exact match is that it *stands out*: the calibrator uses the top similarity plus
its gap to the 2nd and 5th results. Held out on LookBench: when it says >= 90%, the top result
is the exact product 91% of the time (86% at >= 80%).
"""

import json

import numpy as np

from silhouette_vision.config import path

TIERS = [(0.8, "Very likely the exact product"),
         (0.5, "Possibly the exact product"),
         (0.0, "No confident exact match - closest alternatives")]


class MatchConfidence:
    def __init__(self):
        self.params = json.loads((path("models") / "match_confidence.json").read_text())

    def probability(self, top_scores: np.ndarray, mode: str = "marqo") -> float:
        """P(top result is the exact product) from the descending top-5 similarity scores."""
        s = np.sort(np.asarray(top_scores, dtype=np.float64))[::-1][:5]
        x = np.array([s[0], s[0] - s[1], s[0] - s[4]])
        p = self.params[mode]
        z = float(np.dot(p["coef"], x) + p["intercept"])
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def tier(probability: float) -> str:
        return next(label for threshold, label in TIERS if probability >= threshold)
