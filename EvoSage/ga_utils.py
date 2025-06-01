import numpy as np
import random


def _dominates(a, b):
  """Return True if solution ``a`` dominates solution ``b`` (for maximization)."""
  better_or_equal = all(x >= y for x, y in zip(a, b))
  strictly_better = any(x > y for x, y in zip(a, b))
  return better_or_equal and strictly_better


def nsga2_sort(pop, scores):
  """Perform fast non-dominated sorting and compute crowding distance.

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
      if _dominates(scores[p], scores[q]):
        domination_set[p].add(q)
      elif _dominates(scores[q], scores[p]):
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
  contenders = random.sample(pop, k)
  contenders.sort(key=lambda x: (x.get("rank", 0), -x.get("crowding", 0)))
  return contenders[0]["seq"] if isinstance(contenders[0], dict) else contenders[0]


def guided_mutate(seq, cand, p_m=0.08):
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
  for pos, aa_choices in cand.items():
    if pos < 0 or pos >= len(seq_list):
      continue
    if aa_choices and random.random() < p_m:
      seq_list[pos] = random.choice(aa_choices)
  return "".join(seq_list)
