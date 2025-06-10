import numpy as np
import pandas as pd
import random
from typing import Iterable, Sequence
from .prosst_additive import SINGLE_LETTER_CODES


def _dominates(a: Sequence[float], b: Sequence[float], epsilon: float | Iterable[float] = 0.0) -> bool:
  """Return True if solution ``a`` epsilon-dominates solution ``b``."""
  if isinstance(epsilon, Iterable):
    eps = list(epsilon)
  else:
    eps = [epsilon] * len(a)
  strictly_better = False
  for x, y, e in zip(a, b, eps):
    if x + e < y:
      return False
    if x + e > y:
      strictly_better = True
  return strictly_better


def nsga2_sort(pop, scores, epsilon: float | Iterable[float] = 0.0):
  """Perform fast non-dominated sorting and compute crowding distance.

  The optional ``epsilon`` parameter enables epsilon-dominance: solution ``a``
  dominates ``b`` only if ``a_i + epsilon >= b_i`` for all objectives and
  ``a_i + epsilon > b_i`` for at least one objective.

  Parameters
  ----------
  pop : list
    Population of candidate solutions.
  scores : list of tuple[float]
    Objective values corresponding to ``pop``.

  Returns
  -------
  list[list[dict]]
    A list of Pareto fronts. Each front is a list of dictionaries with keys
    ``seq`` (the candidate), ``score`` (objective values), ``rank`` and
    ``crowding`` (crowding distance).
  """
  n = len(pop)
  domination_set = [set() for _ in range(n)]
  dominated_count = [0] * n
  fronts = [[]]

  for p in range(n):
    for q in range(n):
      if p == q:
        continue
      if _dominates(scores[p], scores[q], epsilon):
        domination_set[p].add(q)
      elif _dominates(scores[q], scores[p], epsilon):
        dominated_count[p] += 1
    if dominated_count[p] == 0:
      fronts[0].append(p)

  i = 0
  while fronts[i]:
    next_front = []
    for p in fronts[i]:
      for q in domination_set[p]:
        dominated_count[q] -= 1
        if dominated_count[q] == 0:
          next_front.append(q)
    i += 1
    fronts.append(next_front)
  fronts.pop()  # remove last empty front

  result = []
  num_obj = len(scores[0]) if scores else 0

  for f_idx, f in enumerate(fronts):
    if not f:
      continue
    f_scores = np.array([scores[idx] for idx in f], dtype=float)
    distances = np.zeros(len(f), dtype=float)
    for m in range(num_obj):
      values = f_scores[:, m]
      order = np.argsort(values)
      distances[order[0]] = np.inf
      distances[order[-1]] = np.inf
      vmin = values[order[0]]
      vmax = values[order[-1]]
      if vmax - vmin == 0:
        continue
      for j in range(1, len(f) - 1):
        prev_val = values[order[j - 1]]
        next_val = values[order[j + 1]]
        distances[order[j]] += (next_val - prev_val) / (vmax - vmin)

    front_list = []
    for idx, dist in zip(f, distances):
      front_list.append({
        "seq": pop[idx],
        "score": scores[idx],
        "rank": f_idx,
        "crowding": float(dist),
      })
    result.append(front_list)

  return result


def elitist_selection(fronts, keep=100):
  """Keep top candidates from the Pareto fronts based on crowding distance."""
  new_pop = []
  for front in fronts:
    sorted_front = sorted(front, key=lambda x: x["crowding"], reverse=True)
    for cand in sorted_front:
      if len(new_pop) >= keep:
        return new_pop
      new_pop.append(cand)
  return new_pop


def _hamming(a, b):
  return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def population_diversity(pop):
  """Return mean pairwise Hamming distance normalised by sequence length."""
  if not pop:
    return 0.0
  n = len(pop)
  length = len(pop[0])
  total = 0
  count = 0
  for i in range(n):
    for j in range(i + 1, n):
      total += _hamming(pop[i], pop[j])
      count += 1
  return (total / count / length) if count else 0.0


def enforce_diversity(pop, d_min=5):
  """Greedy diversity filter based on Hamming distance."""
  selected = []
  for cand in pop:
    seq = cand["seq"] if isinstance(cand, dict) else cand
    if all(_hamming(seq, (s["seq"] if isinstance(s, dict) else s)) >= d_min for s in selected):
      selected.append(cand)
  return selected


def tournament(pop, k=3):
  """Perform tournament selection on the population."""
  k = min(k, len(pop))  # Ensure k is not larger than population size
  if k == 0:
    return None # Or raise a more specific error, depending on desired behavior
  contenders = random.sample(pop, k)
  contenders.sort(key=lambda x: (x.get("rank", 0), -x.get("crowding", 0)))
  return contenders[0]["seq"] if isinstance(contenders[0], dict) else contenders[0]


def crossover(parent1, parent2):
  """Return a child produced by uniform crossover of two parent sequences."""
  if len(parent1) != len(parent2):
    raise ValueError("Parents must be the same length")
  child = [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]
  return "".join(child)


def rank_negative_sites(scores: pd.DataFrame) -> list[int]:
  """Return positions ranked by count and sum of negative ProSST scores.

  Parameters
  ----------
  scores : pandas.DataFrame
    ProSST score matrix returned by :func:`run_prosst`.

  Returns
  -------
  list[int]
    0-indexed positions sorted from least to most negative.
  """
  aa_cols = [c for c in scores.columns if c in SINGLE_LETTER_CODES]
  stats = []
  for row in scores.itertuples(index=False):
    pos = int(row.index) - 1
    values = np.array([getattr(row, aa) for aa in aa_cols], dtype=float)
    neg_mask = values < 0
    neg_count = int(neg_mask.sum())
    neg_sum = float(values[neg_mask].sum()) if neg_count > 0 else 0.0
    stats.append((pos, neg_count, neg_sum))
  stats.sort(key=lambda x: (x[1], -x[2]))
  return [s[0] for s in stats]


def guided_mutate(seq, cand, p_m=0.08, fallback_rank=None, weights=None):
  """Mutate ``seq`` only at positions defined in ``cand``.

  Parameters
  ----------
  seq : str
    Input amino-acid sequence.
  cand : dict[int, list[str]]
    Mapping of 0-indexed positions to allowed amino acids.
  p_m : float
    Mutation probability per allowed position.
  """
  seq_list = list(seq)
  mutated = False
  for pos, aa_choices in cand.items():
    if pos < 0 or pos >= len(seq_list):
      continue
    prob = p_m
    if weights:
      prob = p_m * float(weights.get(pos, 1.0))
    if aa_choices and random.random() < prob:
      seq_list[pos] = random.choice(aa_choices)
      mutated = True

  if not cand or not mutated:
    if fallback_rank:
      top_n = max(1, len(fallback_rank) // 4)
      pos = random.choice(fallback_rank[:top_n])
    else:
      pos = random.randrange(len(seq_list))
    fallback_choices = [aa for aa in SINGLE_LETTER_CODES if aa != seq_list[pos]]
    seq_list[pos] = random.choice(fallback_choices)

  return "".join(seq_list)


def _hypervolume(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
  """Return dominated hypervolume for maximization objectives."""
  def hv_recursive(pts: list[list[float]], r: list[float]) -> float:
    dim = len(r)
    if dim == 1:
      return r[0] - min(p[0] for p in pts)
    pts.sort(key=lambda x: x[0])
    hv = 0.0
    prev = r[0]
    while pts:
      x = pts[-1][0]
      slice_pts = [p[1:] for p in pts]
      hv += hv_recursive(slice_pts, r[1:]) * (prev - x)
      prev = x
      pts = [p for p in pts if p[0] < x]
    return hv
  filt = [list(p) for p in points if all(p[i] <= ref[i] for i in range(len(ref)))]
  if not filt:
    return 0.0
  return hv_recursive(filt, list(ref))


def _hv_contributions(points: Sequence[Sequence[float]], ref: Sequence[float]) -> list[float]:
  total = _hypervolume(points, ref)
  contrib = []
  for i in range(len(points)):
    subset = list(points[:i]) + list(points[i+1:])
    contrib.append(total - _hypervolume(subset, ref))
  return contrib


def hypervolume_selection(fronts, keep=100, reference_point=None):
  """Select individuals based on hypervolume contribution."""
  if not fronts:
    return []
  all_scores = [cand["score"] for f in fronts for cand in f]
  num_obj = len(all_scores[0]) if all_scores else 0
  if reference_point is None:
    reference_point = [min(s[i] for s in all_scores) - 1.0 for i in range(num_obj)]

  new_pop = []
  for front in fronts:
    if len(new_pop) + len(front) <= keep:
      new_pop.extend(front)
      continue
    k = keep - len(new_pop)
    scores = [cand["score"] for cand in front]
    contrib = _hv_contributions(scores, reference_point)
    order = sorted(range(len(front)), key=lambda i: contrib[i], reverse=True)
    for idx in order[:k]:
      new_pop.append(front[idx])
    break
  return new_pop
