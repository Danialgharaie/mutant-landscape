import argparse
import os
import random
import tempfile
from typing import Dict, List, Tuple

import pandas as pd

from .prosst_additive import run_prosst, SINGLE_LETTER_CODES
from .ga_utils import (
    nsga2_sort,
    elitist_selection,
    enforce_diversity,
    tournament,
    guided_mutate,
)
from .evoef2 import build_mutant
from .eval import run_destress


def compute_additive_score(seq: str, wt_seq: str, scores: pd.DataFrame) -> float:
    """Compute additive score for ``seq`` using per-position mutation scores."""
    total = 0.0
    for i, (wt, aa) in enumerate(zip(wt_seq, seq), start=1):
        if aa == wt:
            continue
        val = scores.loc[scores["index"] == i, aa]
        if not val.empty:
            total += float(val.values[0])
    return float(total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoSage evolutionary search")
    parser.add_argument("wt_seq", help="Wild-type amino acid sequence")
    parser.add_argument("pdb", help="Path to wild-type PDB file")
    parser.add_argument("--pop_size", type=int, default=50, help="Population size")
    parser.add_argument("--max_k", type=int, default=4, help="Maximum number of simultaneous mutations")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations to run")
    parser.add_argument("--beneficial_th", type=float, default=0.5, help="Threshold for beneficial single mutants")
    parser.add_argument("--neutral_th", type=float, default=0.0, help="Threshold for nearly-neutral single mutants")
    parser.add_argument("--output_csv", help="Optional CSV output path")
    return parser.parse_args()


def _allowed_mutations(scores: pd.DataFrame, neutral_th: float) -> Dict[int, List[str]]:
    allowed: Dict[int, List[str]] = {}
    for row in scores.itertuples(index=False):
        pos = int(row.index) - 1
        aa_list = []
        for aa in SINGLE_LETTER_CODES:
            if getattr(row, aa) >= neutral_th:
                aa_list.append(aa)
        if aa_list:
            allowed[pos] = aa_list
    return allowed


def _good_single_mutants(scores: pd.DataFrame, wt_seq: str, thr: float) -> List[str]:
    seqs = []
    for row in scores.itertuples(index=False):
        pos = int(row.index) - 1
        for aa in SINGLE_LETTER_CODES:
            score = getattr(row, aa)
            if score >= thr and aa != wt_seq[pos]:
                seq_list = list(wt_seq)
                seq_list[pos] = aa
                seqs.append("".join(seq_list))
    return seqs


def _rand_combination(wt_seq: str, allowed: Dict[int, List[str]], max_k: int) -> str:
    seq = list(wt_seq)
    num_mut = random.randint(1, max_k)
    positions = random.sample(list(allowed.keys()), min(num_mut, len(allowed)))
    for p in positions:
        seq[p] = random.choice(allowed[p])
    return "".join(seq)


def _mutfile_from_seq(seq: str, wt_seq: str, chain: str = "A") -> str:
    parts = []
    for i, (wt, aa) in enumerate(zip(wt_seq, seq), start=1):
        if wt != aa:
            parts.append(f"{chain}{wt}{i}{aa}")
    return ",".join(parts) + ";" if parts else ""


def main() -> None:
    args = parse_args()

    wt_seq = args.wt_seq
    pdb = args.pdb

    scores = run_prosst(wt_seq, pdb)
    allowed = _allowed_mutations(scores, args.neutral_th)
    pop: List[str] = [wt_seq]
    pop.extend(_good_single_mutants(scores, wt_seq, args.beneficial_th))
    while len(pop) < args.pop_size:
        pop.append(_rand_combination(wt_seq, allowed, args.max_k))

    destress_cache: Dict[str, float] = {}
    history = []

    for gen in range(args.generations):
        score_list: List[Tuple[float, float]] = []
        for seq in pop:
            add_score = compute_additive_score(seq, wt_seq, scores)
            if seq not in destress_cache:
                mut_str = _mutfile_from_seq(seq, wt_seq)
                if mut_str:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        mut_file = os.path.join(tmpdir, "mut.txt")
                        with open(mut_file, "w") as fh:
                            fh.write(mut_str)
                        mut_pdb = build_mutant(pdb, mut_file, tmpdir)
                        metrics = run_destress(mut_pdb, tmpdir)
                        destress_cache[seq] = -float(metrics["evoef2"]["total"])
                else:
                    destress_cache[seq] = 0.0
            destress_score = destress_cache[seq]
            score_list.append((add_score, destress_score))
            history.append({"seq": seq, "additive": add_score, "destress": destress_score, "gen": gen})

        fronts = nsga2_sort(pop, score_list)
        elite = elitist_selection(fronts, keep=args.pop_size)
        elite = enforce_diversity(elite)
        new_pop = [cand["seq"] for cand in elite]
        while len(new_pop) < args.pop_size:
            parent = tournament(elite)
            child = guided_mutate(parent, allowed)
            new_pop.append(child)
        pop = new_pop

    final_fronts = nsga2_sort(pop, [(compute_additive_score(s, wt_seq, scores), destress_cache.get(s, 0.0)) for s in pop])
    best_front = final_fronts[0]

    print("Final Pareto Front:")
    for cand in best_front:
        seq = cand["seq"]
        add = compute_additive_score(seq, wt_seq, scores)
        dest = destress_cache.get(seq, 0.0)
        print(seq, f"Additive: {add:.3f}", f"Destress: {dest:.3f}")

    if args.output_csv:
        import csv

        with open(args.output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["seq", "additive", "destress", "gen"])
            writer.writeheader()
            writer.writerows(history)


if __name__ == "__main__":
    main()
