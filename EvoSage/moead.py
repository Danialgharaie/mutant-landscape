from itertools import product
import numpy as np
from .ga_utils import nsga2_sort, elitist_selection


def _simplex_lattice(num_obj: int, divisions: int) -> list[list[float]]:
    weights = []
    for coords in product(range(divisions + 1), repeat=num_obj):
        if sum(coords) == divisions:
            weights.append([c / divisions for c in coords])
    return weights


def moead_select(pop, scores, keep=100, divisions=4):
    """Select individuals using a basic MOEA/D decomposition."""
    if not pop:
        return []
    num_obj = len(scores[0]) if scores else 0
    weights = _simplex_lattice(num_obj, divisions)
    if not weights:
        return []
    ref = np.max(np.array(scores), axis=0)
    best_map: dict[int, dict] = {}
    for w in weights:
        best_idx = None
        best_val = np.inf
        for idx, s in enumerate(scores):
            val = max(w[i] * abs(ref[i] - s[i]) for i in range(num_obj))
            if val < best_val:
                best_val = val
                best_idx = idx
        if best_idx is not None:
            cand = {
                "seq": pop[best_idx],
                "score": scores[best_idx],
                "rank": 0,
                "crowding": 0.0,
            }
            best_map[best_idx] = cand
        if len(best_map) >= keep:
            break
    selected = list(best_map.values())
    if len(selected) < keep:
        fronts = nsga2_sort(pop, scores)
        extra = elitist_selection(fronts, keep=keep)
        for cand in extra:
            if cand["seq"] not in {c["seq"] for c in selected}:
                selected.append(cand)
            if len(selected) >= keep:
                break
    return selected[:keep]
