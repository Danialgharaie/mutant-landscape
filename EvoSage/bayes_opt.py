from __future__ import annotations

from typing import Callable, Dict, Tuple

from skopt import gp_minimize
from skopt.space import Integer, Real


def bayesian_optimize(
    evaluate_func: Callable[..., float],
    bounds: Dict[str, Tuple[float, float, str]],
    n_calls: int = 10,
) -> Dict[str, float]:
    """Tune GA hyperparameters using Bayesian optimization.

    Parameters
    ----------
    evaluate_func : callable
        Function that accepts keyword arguments for each hyperparameter and
        returns a numeric score to maximize.
    bounds : dict[str, tuple]
        Mapping of parameter name to a tuple ``(low, high, kind)`` where
        ``kind`` is ``"int"`` or ``"real"``.
    n_calls : int, optional
        Number of optimization iterations, by default 10.

    Returns
    -------
    dict[str, float]
        Mapping of parameter names to the best found values.
    """

    space = []
    for name, (low, high, kind) in bounds.items():
        if kind == "int":
            space.append(Integer(int(low), int(high), name=name))
        else:
            space.append(Real(float(low), float(high), name=name))

    def objective(vals):
        params = {space[i].name: vals[i] for i in range(len(space))}
        score = evaluate_func(**params)
        return -float(score)

    result = gp_minimize(objective, space, n_calls=n_calls, random_state=0)
    best = {space[i].name: result.x[i] for i in range(len(space))}
    return best
