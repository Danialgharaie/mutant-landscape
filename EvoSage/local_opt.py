"""Local optimization utilities for EvoSage.

This module provides a simple hill climbing optimizer
that maximizes the additive ProSST score by exploring
single point mutations at allowed positions.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .scoring import compute_additive_score
from .prosst_additive import SINGLE_LETTER_CODES


def local_optimize(
    seq: str,
    allowed: Dict[int, List[str]] | None,
    score_matrix: pd.DataFrame,
    max_steps: int = 5,
) -> Tuple[str, float]:
    """Perform a simple hill climb on ``seq``.

    Parameters
    ----------
    seq : str
        Starting amino-acid sequence.
    allowed : dict[int, list[str]] or None
        Mapping of 0-indexed positions to allowed residues. ``None``
        means all positions and amino acids are considered.
    score_matrix : pandas.DataFrame
        ProSST score matrix used for additive scoring.
    max_steps : int, optional
        Maximum number of optimization passes, by default 5.

    Returns
    -------
    tuple[str, float]
        The optimized sequence and its additive score.
    """

    best_seq = list(seq)
    best_score = compute_additive_score(seq, score_matrix)

    positions: Iterable[int]
    if allowed:
        positions = list(allowed.keys())
    else:
        positions = range(len(best_seq))

    for _ in range(max_steps):
        improved = False
        for pos in positions:
            aa_choices = allowed.get(pos) if allowed else SINGLE_LETTER_CODES
            for aa in aa_choices:
                if aa == best_seq[pos]:
                    continue
                cand = best_seq.copy()
                cand[pos] = aa
                cand_seq = "".join(cand)
                score = compute_additive_score(cand_seq, score_matrix)
                if score > best_score:
                    best_seq = cand
                    best_score = score
                    improved = True
        if not improved:
            break

    return "".join(best_seq), float(best_score)
