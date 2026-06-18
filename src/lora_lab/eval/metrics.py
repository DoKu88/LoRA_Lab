"""SNI eval metrics: exact-match and ROUGE-L.

Both score a prediction against a *list* of acceptable references (SNI outputs
are multi-reference) by taking the max, then average over examples. This
matches the Super-Natural-Instructions convention (EM for classification,
ROUGE-L for free-form generation).
"""

from __future__ import annotations

import re
import string


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles/punct/extra space."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    pred = normalize(prediction)
    return 1.0 if any(pred == normalize(r) for r in references) else 0.0


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_t = normalize(prediction).split()
    ref_t = normalize(reference).split()
    lcs = _lcs_len(pred_t, ref_t)
    if lcs == 0 or not pred_t or not ref_t:
        return 0.0
    prec = lcs / len(pred_t)
    rec = lcs / len(ref_t)
    return 2 * prec * rec / (prec + rec)


def rouge_l(prediction: str, references: list[str]) -> float:
    """Max ROUGE-L F1 over references (self-contained LCS implementation)."""
    return max((_rouge_l_f1(prediction, r) for r in references), default=0.0)


_METRICS = {"exact_match": exact_match, "rougeL": rouge_l}


def score_predictions(
    predictions: list[str],
    references: list[list[str]],
    metric: str,
) -> dict[str, float]:
    """Average the metric over examples. Returns ``{metric, n, sum}``."""
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; have {list(_METRICS)}")
    fn = _METRICS[metric]
    if len(predictions) != len(references):
        raise ValueError("predictions and references length mismatch")
    scores = [fn(p, r) for p, r in zip(predictions, references)]
    n = len(scores)
    total = sum(scores)
    return {"metric": metric, "score": (total / n) if n else 0.0, "n": n}
