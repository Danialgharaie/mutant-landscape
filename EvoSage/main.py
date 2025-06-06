import argparse
import os
import random
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import datetime
import numpy as np
from tqdm.auto import tqdm

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
from .scoring import compute_additive_score
from . import logger, setup_logging

def ascii_splash():
    """Returns an ASCII art splash screen for EvoSage."""
    ascii_art = """


░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░ ░▒▓███████▓▒░░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓████████▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░       ░▒▓█▓▒▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░        
░▒▓██████▓▒░  ░▒▓█▓▒▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒▒▓███▓▒░▒▓██████▓▒░   
░▒▓█▓▒░        ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░        ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓████████▓▒░  ░▒▓██▓▒░   ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░ 
                                                                                             
                                                                                             

"""
    print(ascii_art)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoSage evolutionary search")
    parser.add_argument("wt_seq", help="Wild-type amino acid sequence")
    parser.add_argument("pdb", help="Path to wild-type PDB file")
    parser.add_argument("--pop_size", type=int, default=50, help="Population size")
    parser.add_argument("--max_k", type=int, default=4, help="Maximum number of simultaneous mutations")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations to run")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Generations with no additive improvement before population reset",
    )
    parser.add_argument("--beneficial_th", type=float, default=0.5, help="Threshold for beneficial single mutants")
    parser.add_argument("--neutral_th", type=float, default=0.0, help="Threshold for nearly-neutral single mutants")
    parser.add_argument("--output_csv", help="Optional CSV output path")
    parser.add_argument("--out_dir", help="Directory to store per-generation results")
    parser.add_argument(
        "--dynamic_prosst",
        action="store_true",
        help="Recompute ProSST score matrix using best sequence each generation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
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
    if not allowed:
        pos = random.randrange(len(seq))
        aa_choices = [aa for aa in SINGLE_LETTER_CODES if aa != wt_seq[pos]]
        seq[pos] = random.choice(aa_choices)
        return "".join(seq)

    num_mut = random.randint(1, max_k)
    positions = random.sample(list(allowed.keys()), min(num_mut, len(allowed)))
    for p in positions:
        seq[p] = random.choice(allowed[p])
    return "".join(seq)


def _mutfile_from_seq(seq: str, wt_seq: str, chain: str = "A") -> str:
    parts = []
    for i, (wt, aa) in enumerate(zip(wt_seq, seq), start=1):
        if wt != aa:
            parts.append(f"{wt}{chain}{i}{aa}")
    return ",".join(parts) + ";" if parts else ""


def _extract_evoef2_total(metrics: Dict[str, Any]) -> float:
    """Return the EvoEF2 total energy value from a metrics dictionary."""
    for key, val in metrics.items():
        if "evoef2" in key.lower() and "total" in key.lower():
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    raise KeyError("EvoEF2 total energy not found in metrics")


METRIC_INFO = {
    # substrings to identify the metric in De-StReSS CSV, improvement direction
    "aggrescan_avg": {"keys": ["aggrescan", "avg"], "direction": "down"},
    "aggrescan_min": {"keys": ["aggrescan", "min"], "direction": "down"},
    "aggrescan_max": {"keys": ["aggrescan", "max"], "direction": "down"},
    "bude_total": {"keys": ["bude", "total"], "direction": "down"},
    "dfire2_total": {"keys": ["dfire2", "total"], "direction": "down"},
    "evoef2_total": {"keys": ["evoef2", "total"], "direction": "down"},
    "rosetta_total": {"keys": ["rosetta", "total"], "direction": "down"},
    "bude_steric": {"keys": ["bude", "steric"], "direction": "down"},
    "bude_desolvation": {"keys": ["bude", "desolvation"], "direction": "down"},
    "bude_charge": {"keys": ["bude", "charge"], "direction": "down"},
    "rosetta_fa_atr": {"keys": ["rosetta", "fa_atr"], "direction": "down"},
    "rosetta_fa_rep": {"keys": ["rosetta", "fa_rep"], "direction": "down"},
    "packing_density": {"keys": ["packing", "density"], "direction": "up"},
    "hydrophobic_fitness": {"keys": ["hydrophobic", "fitness"], "direction": "up"},
    "evoef2_intraR_total": {"keys": ["evoef2", "intrar"], "direction": "down"},
    "evoef2_interS_total": {"keys": ["evoef2", "inters"], "direction": "down"},
}

STABILITY_METRICS = ["evoef2_total", "bude_total", "dfire2_total", "rosetta_total"]
CORE_QUALITY_METRICS = ["packing_density", "hydrophobic_fitness", "bude_desolvation"]
SOLUBILITY_METRICS = ["aggrescan_avg", "aggrescan_min", "aggrescan_max"]


def _find_metric(metrics: Dict[str, Any], substrings: List[str]) -> float:
    key_low_map = {k.lower().replace(" ", ""): k for k in metrics}
    for k_low, original in key_low_map.items():
        if all(s.lower() in k_low for s in substrings):
            try:
                return float(metrics[original])
            except (TypeError, ValueError):
                break
    raise KeyError(f"metric {' '.join(substrings)} not found")


def compute_deltas(mut_metrics: Dict[str, Any], orig_metrics: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    deltas: Dict[str, float] = {}
    scores: Dict[str, float] = {}
    for name, info in METRIC_INFO.items():
        try:
            mut_val = _find_metric(mut_metrics, info["keys"])
            orig_val = _find_metric(orig_metrics, info["keys"])
        except KeyError:
            continue
        delta = mut_val - orig_val
        deltas[f"delta_{name}"] = delta
        if info["direction"] == "down":
            scores[f"score_{name}"] = -delta
        else:
            scores[f"score_{name}"] = delta
    return deltas, scores


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    wt_seq = args.wt_seq
    pdb = args.pdb

    ascii_splash()

    logger.info(
        "Starting EvoSage run: pop_size=%d, generations=%d",
        args.pop_size,
        args.generations,
    )

    scores = run_prosst(wt_seq, pdb)
    allowed = _allowed_mutations(scores, args.neutral_th)
    if not allowed:
        allowed = {
            i: [aa for aa in SINGLE_LETTER_CODES if aa != wt_seq[i]]
            for i in range(len(wt_seq))
        }
    pop: List[str] = []
    seen_global: set[str] = {wt_seq}

    for seq in _good_single_mutants(scores, wt_seq, args.beneficial_th):
        if seq != wt_seq and seq not in seen_global:
            pop.append(seq)
            seen_global.add(seq)

    while len(pop) < args.pop_size:
        cand = _rand_combination(wt_seq, allowed, args.max_k)
        if cand != wt_seq and cand not in seen_global:
            pop.append(cand)
            seen_global.add(cand)

    run_root = args.out_dir or f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    Path(run_root).mkdir(parents=True, exist_ok=True)

    # generation 0 folder with wild-type files and metrics
    gen0_dir = Path(run_root) / "generation_00"
    (gen0_dir / "mutants").mkdir(parents=True, exist_ok=True)
    wt_pdb_out = gen0_dir / "wildtype.pdb"
    shutil.copy(pdb, wt_pdb_out)
    with tempfile.TemporaryDirectory() as tmpdir:
        wt_tmp = os.path.join(tmpdir, "wt.pdb")
        shutil.copy(pdb, wt_tmp)
        wt_metrics_dict = run_destress(tmpdir)
        csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(tmpdir, x)))
            shutil.move(os.path.join(tmpdir, latest_csv), gen0_dir / "wildtype_destress.csv")
        orig_metrics = next(iter(wt_metrics_dict.values()))

    destress_cache: Dict[str, Dict[str, Any]] = {}
    delta_zero, score_zero = compute_deltas(orig_metrics, orig_metrics)
    destress_cache[wt_seq] = {
        "delta": delta_zero,
        "score": score_zero,
        "pdb_path": str(wt_pdb_out),
        "csv_path": str(gen0_dir / "wildtype_destress.csv"),
    }

    history: List[Dict[str, Any]] = []
    final_df: pd.DataFrame | None = None

    best_overall_seq = wt_seq
    best_overall_score = compute_additive_score(wt_seq, scores)
    stale_count = 0

    for gen in tqdm(range(args.generations), desc="Generation"):
        logger.info("Starting generation %d", gen)
        pop = [seq for i, seq in enumerate(pop) if seq != wt_seq and seq not in pop[:i]]
        seen_global.update(pop)
        gen_dir = Path(run_root) / f"generation_{gen:02d}"
        mutants_dir = gen_dir / "mutants"
        mutants_dir.mkdir(parents=True, exist_ok=True)

        gen_rows: List[Dict[str, Any]] = []

        for idx, seq in enumerate(pop):
            add_score = compute_additive_score(seq, scores)
            if seq not in destress_cache:
                mut_str = _mutfile_from_seq(seq, wt_seq)
                with tempfile.TemporaryDirectory() as tmpdir:
                    if mut_str and mut_str != ";":
                        logger.debug(mut_str)
                        mut_file = os.path.join(tmpdir, "mut.txt")
                        with open(mut_file, "w") as fh:
                            fh.write(mut_str)
                        mut_pdb = build_mutant(pdb, mut_file, tmpdir, quiet=True)
                    else:
                        shutil.copy(pdb, os.path.join(tmpdir, "model.pdb"))
                        mut_pdb = os.path.join(tmpdir, "model.pdb")
                    metrics_dict = run_destress(tmpdir)
                    csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
                    csv_src = max(csv_files, key=lambda x: os.path.getctime(os.path.join(tmpdir, x)))
                    dest_pdb = mutants_dir / f"mut_{idx:04d}.pdb"
                    dest_csv = mutants_dir / f"mut_{idx:04d}_destress.csv"
                    shutil.move(mut_pdb, dest_pdb)
                    shutil.move(os.path.join(tmpdir, csv_src), dest_csv)
                    mut_metrics = next(iter(metrics_dict.values()))
                deltas, scores_dict = compute_deltas(mut_metrics, orig_metrics)
                destress_cache[seq] = {
                    "delta": deltas,
                    "score": scores_dict,
                    "pdb_path": str(dest_pdb),
                    "csv_path": str(dest_csv),
                }
            else:
                data = destress_cache[seq]
                dest_pdb = mutants_dir / f"mut_{idx:04d}.pdb"
                dest_csv = mutants_dir / f"mut_{idx:04d}_destress.csv"
                shutil.copy(data["pdb_path"], dest_pdb)
                shutil.copy(data["csv_path"], dest_csv)

            entry = {
                "seq": seq,
                "additive": add_score,
                **destress_cache[seq]["delta"],
                **destress_cache[seq]["score"],
            }
            gen_rows.append(entry)

        df = pd.DataFrame(gen_rows).fillna(0.0)
        for name in METRIC_INFO:
            score_col = f"score_{name}"
            if score_col not in df:
                df[score_col] = 0.0
            mean = df[score_col].mean()
            std = df[score_col].std(ddof=0)
            df[f"{score_col}_z"] = (df[score_col] - mean) / std if std > 0 else 0.0

        df["Stability_z"] = df[[f"score_{m}_z" for m in STABILITY_METRICS]].mean(axis=1)
        df["CoreQuality_z"] = df[[f"score_{m}_z" for m in CORE_QUALITY_METRICS]].mean(axis=1)
        df["Solubility_z"] = df[[f"score_{m}_z" for m in SOLUBILITY_METRICS]].mean(axis=1)

        df.insert(0, "gen", gen)
        df.to_csv(gen_dir / "metrics_with_delta.csv", index=False)

        # Save sequences for this generation to a FASTA file
        fasta_path = gen_dir / "population.fasta"
        with open(fasta_path, "w") as fh:
            for i, seq in enumerate(df["seq"], start=1):
                fh.write(f">seq{i}\n{seq}\n")

        final_df = df

        score_list = []
        for row in df.itertuples(index=False):
            seq = row.seq
            destress_cache[seq]["z"] = {
                "Stability_z": row.Stability_z,
                "CoreQuality_z": row.CoreQuality_z,
                "Solubility_z": row.Solubility_z,
            }
            score_list.append(
                (
                    row.additive,
                    -row.Stability_z,
                    -row.CoreQuality_z,
                    -row.Solubility_z,
                )
            )
            history.append(row._asdict())

        fronts = nsga2_sort(pop, score_list)
        elite = elitist_selection(fronts, keep=args.pop_size)
        elite = enforce_diversity(elite)
        new_pop: List[str] = []
        next_seen: set[str] = set()
        for cand in elite:
            seq = cand["seq"]
            if seq == wt_seq or seq in seen_global or seq in next_seen:
                continue
            new_pop.append(seq)
            next_seen.add(seq)

        attempts = 0
        max_attempts = args.pop_size * 10
        while len(new_pop) < args.pop_size and attempts < max_attempts:
            parent = tournament(elite)
            if parent is None:
                break
            child = guided_mutate(parent, allowed)
            attempts += 1
            if child == wt_seq or child in seen_global or child in next_seen:
                continue
            logger.debug("Selected candidate %s", child)
            new_pop.append(child)
            next_seen.add(child)

        while len(new_pop) < args.pop_size:
            cand = _rand_combination(wt_seq, allowed, args.max_k)
            if cand == wt_seq or cand in seen_global or cand in next_seen:
                continue
            new_pop.append(cand)
            next_seen.add(cand)

        pop = new_pop
        seen_global.update(pop)

        best_idx = df['additive'].idxmax()
        best_row = df.loc[best_idx]
        logger.info(
            "Generation %d summary: pop=%d unique=%d best_seq=%s add=%.3f Stab_z=%.3f Core_z=%.3f Sol_z=%.3f",
            gen,
            len(df),
            df['seq'].nunique(),
            best_row.seq,
            best_row.additive,
            best_row.Stability_z,
            best_row.CoreQuality_z,
            best_row.Solubility_z,
        )

        if best_row.additive > best_overall_score:
            best_overall_score = best_row.additive
            best_overall_seq = best_row.seq
            stale_count = 0
        else:
            stale_count += 1

        if stale_count >= args.patience:
            logger.warning(
                "No additive improvement for %d generations. Resetting population around %s",
                stale_count,
                best_overall_seq,
            )
            stale_count = 0
            pdb_path = destress_cache[best_overall_seq]["pdb_path"]
            new_scores = run_prosst(best_overall_seq, pdb_path)
            allowed = _allowed_mutations(new_scores, args.neutral_th)
            if not allowed:
                allowed = {
                    i: [aa for aa in SINGLE_LETTER_CODES if aa != best_overall_seq[i]]
                    for i in range(len(best_overall_seq))
                }
            scores = new_scores
            pop = [best_overall_seq]
            seen_global.add(best_overall_seq)
            while len(pop) < args.pop_size:
                cand = _rand_combination(best_overall_seq, allowed, args.max_k)
                if cand == best_overall_seq or cand in seen_global or cand in pop:
                    continue
                pop.append(cand)
                seen_global.add(cand)

        logger.info("Finished generation %d", gen)

        if args.dynamic_prosst and fronts and fronts[0]:
            best_seq = fronts[0][0]["seq"]
            logger.info("Updating ProSST using best sequence %s", best_seq)
            pdb_path = destress_cache[best_seq]["pdb_path"]
            scores = run_prosst(best_seq, pdb_path)
            allowed = _allowed_mutations(scores, args.neutral_th)
            if not allowed:
                allowed = {
                    i: [aa for aa in SINGLE_LETTER_CODES if aa != best_seq[i]]
                    for i in range(len(best_seq))
                }

    if final_df is not None:
        final_fronts = nsga2_sort(
            final_df["seq"].tolist(),
            list(
                zip(
                    final_df["additive"],
                    -final_df["Stability_z"],
                    -final_df["CoreQuality_z"],
                    -final_df["Solubility_z"],
                )
            ),
        )
        best_front = final_fronts[0]

        logger.info("Final Pareto Front:")
        seen: set[str] = set()  # filter out duplicate sequences
        for cand in best_front:
            seq = cand["seq"]
            if seq in seen:
                continue
            seen.add(seq)
            add = compute_additive_score(seq, scores)
            z = destress_cache[seq]["z"]
            logger.info(
                "%s Additive: %.3f Stab_z: %.3f Core_z: %.3f Sol_z: %.3f",
                seq,
                add,
                z['Stability_z'],
                z['CoreQuality_z'],
                z['Solubility_z'],
            )

    if args.output_csv:
        seen_csv: set[str] = set()  # remove duplicate sequences
        unique_rows = []
        for row in history:
            seq = row["seq"]
            if seq in seen_csv:
                continue
            seen_csv.add(seq)
            unique_rows.append(row)
        history_df = pd.DataFrame(unique_rows)
        history_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
