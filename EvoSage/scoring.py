"""Utility functions for working with ProSST score matrices."""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import pandas as pd

__all__ = ["compute_additive_score"]


Mutation = Tuple[int, str]


def compute_additive_score(seq: Union[str, Iterable[Mutation]], score_matrix: pd.DataFrame) -> float:
  """Compute the additive ProSST score for a mutant.

  Parameters
  ----------
  seq : str or iterable of (int, str)
    Mutant amino-acid sequence or list of ``(position, residue)`` tuples.
    Positions are 1-indexed.
  score_matrix : pandas.DataFrame
    DataFrame returned by :func:`EvoSage.prosst_additive.run_prosst`.

  Returns
  -------
  float
    The additive mutation score.
  """
  if "index" not in score_matrix.columns or "wt" not in score_matrix.columns:
    raise ValueError("score_matrix must contain 'index' and 'wt' columns")

  df = score_matrix.set_index("index")
  valid_aas = set(df.columns) - {"wt"}

  if isinstance(seq, str):
    if len(seq) != len(df):
      raise ValueError("sequence length does not match score matrix")
    wt_seq = "".join(df["wt"].tolist())
    mutations = [(i, aa) for i, (aa, wt) in enumerate(zip(seq, wt_seq), start=1) if aa != wt]
  else:
    try:
      mutations = list(seq)
    except TypeError as exc:
      raise TypeError("seq must be a string or iterable of (position, aa)") from exc

  score = 0.0
  for pos, aa in mutations:
    if not isinstance(pos, int) or pos < 1 or pos > len(df):
      raise ValueError(f"invalid position: {pos}")
    if aa not in valid_aas:
      raise ValueError(f"invalid amino acid: {aa}")
    score += float(df.at[pos, aa])

  return float(score)
