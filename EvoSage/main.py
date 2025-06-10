import argparse
import json
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
    hypervolume_selection,
    tournament,
    crossover,
    guided_mutate,
    rank_negative_sites,
)
from .moead import moead_select
from .evoef2 import build_mutant
from .eval import run_destress
from .scoring import compute_additive_score
from .local_opt import local_optimize
from .island import AdaptiveGridArchive, cluster_niching
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
    """Parse command line arguments and optional JSON configuration."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        help="Path to JSON configuration file. Values here provide defaults.",
    )

    # Parse only --config first so we know whether to load defaults from a file
    pre_args, _ = pre.parse_known_args()

    config_data: dict[str, Any] = {}
    if pre_args.config:
        with open(pre_args.config) as fh:
            config_data = json.load(fh)

    parser = argparse.ArgumentParser(
        description="EvoSage evolutionary search",
        parents=[pre],
    )
    parser.add_argument(
        "wt_seq",
        nargs="?",
        default=config_data.get("wt_seq"),
        help="Wild-type amino acid sequence",
    )
    parser.add_argument(
        "pdb",
        nargs="?",
        default=config_data.get("pdb"),
        help="Path to wild-type PDB file",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=config_data.get("pop_size", 50),
        help="Population size",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=config_data.get("max_k", 4),
        help="Maximum number of simultaneous mutations",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=config_data.get("generations", 20),
        help="Number of generations to run",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=config_data.get("patience", 10),
        help="Generations with no additive improvement before population reset",
    )
    parser.add_argument(
        "--beneficial_th",
        type=float,
        default=config_data.get("beneficial_th", 0.5),
        help="Threshold for beneficial single mutants",
    )
    parser.add_argument(
        "--neutral_th",
        type=float,
        default=config_data.get("neutral_th", 0.0),
        help="Threshold for nearly-neutral single mutants",
    )
    parser.add_argument(
        "--out_dir",
        default=config_data.get("out_dir"),
        help="Directory to store per-generation results",
    )
    parser.add_argument(
        "--dynamic_prosst",
        action="store_true",
        default=config_data.get("dynamic_prosst", False),
        help="Recompute ProSST score matrix using best sequence each generation",
    )
    parser.add_argument(
        "--mutation_prob",
        type=float,
        default=config_data.get("mutation_prob", 0.08),
        help="Mutation probability per allowed position",
    )
    parser.add_argument(
        "--pm-start",
        type=float,
        dest="pm_start",
        default=config_data.get("pm_start"),
        help="Starting mutation probability for adaptive schedule",
    )
    parser.add_argument(
        "--pm-min",
        type=float,
        dest="pm_min",
        default=config_data.get("pm_min"),
        help="Minimum mutation probability when adapting",
    )
    parser.add_argument(
        "--pm-decay",
        type=float,
        default=config_data.get("pm_decay", 1.0),
        help="Multiplicative decay factor for adaptive mutation probability",
    )
    parser.add_argument(
        "--cr-start",
        type=float,
        dest="cr_start",
        default=config_data.get("cr_start"),
        help="Starting crossover rate for adaptive schedule",
    )
    parser.add_argument(
        "--cr-min",
        type=float,
        dest="cr_min",
        default=config_data.get("cr_min"),
        help="Minimum crossover rate when adapting",
    )
    parser.add_argument(
        "--cr-decay",
        type=float,
        dest="cr_decay",
        default=config_data.get("cr_decay", 1.0),
        help="Multiplicative decay factor for adaptive crossover rate",
    )
    parser.add_argument(
        "--diversity-thresh",
        type=float,
        dest="diversity_thresh",
        default=config_data.get("diversity_thresh", 0.0),
        help="Diversity threshold to trigger mutation rate decay",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config_data.get("epsilon", 0.0),
        help="Epsilon for epsilon-dominance in Pareto sorting",
    )
    parser.add_argument(
        "--hv-selection",
        dest="hv_selection",
        action="store_true",
        default=config_data.get("hv_selection", False),
        help="Use hypervolume-based selection",
    )
    parser.add_argument(
        "--moead-selection",
        dest="moead_selection",
        action="store_true",
        default=config_data.get("moead_selection", False),
        help="Use MOEA/D decomposition-based selection",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=config_data.get("crossover_rate", 0.5),
        help="Probability of applying crossover when generating offspring",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=config_data.get("log_level", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config_data.get("seed"),
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--islands",
        type=int,
        default=config_data.get("islands", 1),
        help="Number of islands for island-model GA",
    )
    parser.add_argument(
        "--migration-interval",
        type=int,
        dest="migration_interval",
        default=config_data.get("migration_interval", 5),
        help="Generations between island migrations",
    )
    parser.add_argument(
        "--migrants",
        type=int,
        default=config_data.get("migrants", 2),
        help="Number of individuals exchanged during migration",
    )
    parser.add_argument(
        "--niche-clusters",
        type=int,
        dest="niche_clusters",
        default=config_data.get("niche_clusters", 5),
        help="Number of clusters for niching",
    )
    parser.add_argument(
        "--niche-size",
        type=int,
        dest="niche_size",
        default=config_data.get("niche_size", 3),
        help="Max individuals kept per niche",
    )
    parser.add_argument(
        "--lamarckian",
        dest="lamarck",
        action="store_true",
        default=config_data.get("lamarck", False),
        help="Replace individuals with their local optimum before evaluation",
    )
    parser.add_argument(
        "--baldwinian",
        dest="baldwin",
        action="store_true",
        default=config_data.get("baldwin", False),
        help="Use local optimum score but keep original sequence",
    )
    parser.add_argument(
        "--opt-steps",
        type=int,
        dest="opt_steps",
        default=config_data.get("opt_steps", 5),
        help="Number of hill-climb steps for local optimization",
    )
    args = parser.parse_args(namespace=argparse.Namespace(**config_data))

    if args.wt_seq is None or args.pdb is None:
        parser.error("wt_seq and pdb must be provided either via CLI or JSON config")

    if args.lamarck and args.baldwin:
        parser.error("--lamarckian and --baldwinian modes are mutually exclusive")

    if args.pm_start is None:
        args.pm_start = args.mutation_prob
    if args.pm_min is None:
        args.pm_min = args.mutation_prob
    if args.cr_start is None:
        args.cr_start = args.crossover_rate
    if args.cr_min is None:
        args.cr_min = args.crossover_rate
    return args


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


def _rand_combination(
    wt_seq: str, allowed: Dict[int, List[str]], max_k: int, fallback_rank: List[int] | None = None
) -> str:
    seq = list(wt_seq)
    if not allowed:
        if fallback_rank:
            top_n = max(1, len(fallback_rank) // 4)
            pos = random.choice(fallback_rank[:top_n])
        else:
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


def _fill_random_unique(
    base_seq: str,
    allowed: Dict[int, List[str]],
    max_k: int,
    fallback_rank: List[int] | None,
    seen: set[str],
    num_needed: int,
    attempt_factor: int = 1000,
) -> list[str]:
    """Generate up to ``num_needed`` unique sequences using ``_rand_combination``.

    This helper keeps trying random combinations while avoiding duplicates and the
    wild type sequence. ``attempt_factor`` controls the number of attempts per
    required sequence to avoid endless loops in degenerate cases.
    """

    result: list[str] = []
    attempts = 0
    max_attempts = num_needed * attempt_factor
    while len(result) < num_needed and attempts < max_attempts:
        cand = _rand_combination(base_seq, allowed, max_k, fallback_rank)
        attempts += 1
        if cand == base_seq or cand in seen or cand in result:
            continue
        result.append(cand)
        seen.add(cand)
    return result


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
    fallback_rank = rank_negative_sites(scores)
    if not allowed:
        allowed = {
            i: [aa for aa in SINGLE_LETTER_CODES if aa != wt_seq[i]]
            for i in range(len(wt_seq))
        }
    populations: list[list[str]] = []
    seen_global: set[str] = {wt_seq}

    base_beneficial = [seq for seq in _good_single_mutants(scores, wt_seq, args.beneficial_th) if seq != wt_seq]
    ben_idx = 0
    for _ in range(args.islands):
        pop: list[str] = []
        while ben_idx < len(base_beneficial) and len(pop) < args.pop_size:
            seq = base_beneficial[ben_idx]
            ben_idx += 1
            if seq not in seen_global:
                pop.append(seq)
                seen_global.add(seq)

        attempts = 0
        max_attempts = args.pop_size * 100
        while len(pop) < args.pop_size and attempts < max_attempts:
            cand = _rand_combination(wt_seq, allowed, args.max_k, fallback_rank)
            attempts += 1
            if cand != wt_seq and cand not in seen_global:
                pop.append(cand)
                seen_global.add(cand)

        if len(pop) < args.pop_size:
            logger.warning(
                "Unable to generate enough unique initial sequences after %d attempts. "
                "Filling remaining population with random unique sequences.",
                attempts,
            )
            needed = args.pop_size - len(pop)
            extra = _fill_random_unique(
                wt_seq,
                allowed,
                args.max_k,
                fallback_rank,
                seen_global,
                needed,
            )
            pop.extend(extra)

        populations.append(pop)

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

    def process_island(pop: list[str], island_idx: int, gen: int) -> tuple[list[str], pd.DataFrame]:
        nonlocal allowed, fallback_rank, scores
        gen_dir = Path(run_root) / f"generation_{gen:02d}_island_{island_idx}"
        mutants_dir = gen_dir / "mutants"
        mutants_dir.mkdir(parents=True, exist_ok=True)

        gen_rows: list[dict[str, Any]] = []
        destress_pending: list[tuple[str, Path, float]] = []

        for idx, seq in enumerate(pop):
            orig_seq = seq
            if args.lamarck or args.baldwin:
                opt_seq, add_score = local_optimize(seq, allowed, scores, args.opt_steps)
                if args.lamarck:
                    seq = opt_seq
                    pop[idx] = seq
            else:
                add_score = compute_additive_score(seq, scores)
            eval_seq = seq if args.lamarck else orig_seq
            dest_pdb = mutants_dir / f"mut_{island_idx}_{idx:04d}.pdb"
            if eval_seq not in destress_cache:
                mut_str = _mutfile_from_seq(eval_seq, wt_seq)
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
                    shutil.move(mut_pdb, dest_pdb)
                destress_pending.append((eval_seq, dest_pdb, add_score))
            else:
                data = destress_cache[eval_seq]
                shutil.copy(data["pdb_path"], dest_pdb)
                entry = {
                    "seq": orig_seq if args.baldwin else seq,
                    "additive": add_score,
                    **destress_cache[eval_seq]["delta"],
                    **destress_cache[eval_seq]["score"],
                }
                gen_rows.append(entry)

        if destress_pending:
            with tempfile.TemporaryDirectory() as tmpdir:
                for _, pdb_path, _ in destress_pending:
                    shutil.copy(pdb_path, os.path.join(tmpdir, os.path.basename(pdb_path)))
                metrics_dict = run_destress(tmpdir)
                csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
                latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(tmpdir, x)))
                csv_src = os.path.join(tmpdir, latest_csv)
                destress_out = gen_dir / "destress.csv"
                shutil.move(csv_src, destress_out)
                for seq, pdb_path, add_score in destress_pending:
                    key = os.path.basename(pdb_path)
                    mut_metrics = metrics_dict.get(key) or metrics_dict.get(os.path.splitext(key)[0])
                    if mut_metrics is None:
                        raise RuntimeError(f"DeStReSS metrics missing for {pdb_path}")
                    deltas, scores_dict = compute_deltas(mut_metrics, orig_metrics)
                    destress_cache[seq] = {
                        "delta": deltas,
                        "score": scores_dict,
                        "pdb_path": str(pdb_path),
                        "csv_path": str(destress_out),
                    }
                    entry = {
                        "seq": seq,
                        "additive": add_score,
                        **deltas,
                        **scores_dict,
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

        fasta_path = gen_dir / "population.fasta"
        with open(fasta_path, "w") as fh:
            for i, seq in enumerate(df["seq"], start=1):
                fh.write(f">seq{i}\n{seq}\n")

        final_dfs[island_idx] = df

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

        fronts = nsga2_sort(pop, score_list, epsilon=args.epsilon)
        if fronts and fronts[0]:
            for cand in fronts[0]:
                archive.add(cand["seq"], cand["score"])

        if args.moead_selection:
            elite = moead_select(pop, score_list, keep=args.pop_size)
        elif args.hv_selection:
            elite = hypervolume_selection(fronts, keep=args.pop_size)
        else:
            elite = elitist_selection(fronts, keep=args.pop_size)

        elite = enforce_diversity(elite)
        elite = cluster_niching(df, elite, args.niche_clusters, args.niche_size)

        new_pop: list[str] = []
        next_seen: set[str] = set()
        for cand in elite:
            seq = cand["seq"]
            if seq == wt_seq or seq in next_seen:
                continue
            new_pop.append(seq)
            next_seen.add(seq)

        attempts = 0
        max_attempts = args.pop_size * 100
        while len(new_pop) < args.pop_size and attempts < max_attempts:
            parent1 = tournament(elite)
            parent2 = tournament(elite)
            if parent1 is None or parent2 is None:
                break
            child = crossover(parent1, parent2) if random.random() < args.crossover_rate else parent1
            child = guided_mutate(child, allowed, p_m=args.mutation_prob, fallback_rank=fallback_rank)
            attempts += 1
            if child == wt_seq or child in next_seen:
                continue
            new_pop.append(child)
            next_seen.add(child)

        if len(new_pop) < args.pop_size:
            needed = args.pop_size - len(new_pop)
            extra = _fill_random_unique(wt_seq, allowed, args.max_k, fallback_rank, next_seen, needed)
            new_pop.extend(extra)
        return new_pop, df

    history: List[Dict[str, Any]] = []
    archive = AdaptiveGridArchive(dim=4)
    final_dfs: list[pd.DataFrame | None] = [None] * args.islands

    for gen in tqdm(range(args.generations), desc="Generation"):
        logger.info("Starting generation %d", gen)
        island_dfs = []
        for isl_idx in range(args.islands):
            populations[isl_idx], df = process_island(populations[isl_idx], isl_idx, gen)
            island_dfs.append(df)

        if args.islands > 1 and gen % args.migration_interval == 0 and gen > 0:
            for i in range(args.islands):
                migrants = island_dfs[i].sort_values("additive", ascending=False)["seq"].head(args.migrants).tolist()
                dest = (i + 1) % args.islands
                dest_df = island_dfs[dest].sort_values("additive")
                worst = dest_df["seq"].head(len(migrants)).tolist()
                populations[dest] = [s for s in populations[dest] if s not in worst] + migrants

    combined_df = pd.concat([df for df in final_dfs if df is not None], ignore_index=True)
    if not combined_df.empty:
        final_fronts = nsga2_sort(
            combined_df["seq"].tolist(),
            list(
                zip(
                    combined_df["additive"],
                    -combined_df["Stability_z"],
                    -combined_df["CoreQuality_z"],
                    -combined_df["Solubility_z"],
                )
            ),
            epsilon=args.epsilon,
        )
        best_front = final_fronts[0]

        logger.info("Final Pareto Front:")
        seen: set[str] = set()
        front_rows: list[dict[str, Any]] = []
        for cand in best_front:
            seq = cand["seq"]
            if seq in seen:
                continue
            seen.add(seq)
            row = combined_df[combined_df["seq"] == seq].iloc[0]
            front_rows.append(
                {
                    "seq": seq,
                    "additive": row.additive,
                    "Stability_z": row.Stability_z,
                    "CoreQuality_z": row.CoreQuality_z,
                    "Solubility_z": row.Solubility_z,
                }
            )
            logger.info(
                "%s Additive: %.3f Stab_z: %.3f Core_z: %.3f Sol_z: %.3f",
                seq,
                row.additive,
                row.Stability_z,
                row.CoreQuality_z,
                row.Solubility_z,
            )

        pd.DataFrame(front_rows).to_csv(Path(run_root) / "final_pareto_front.csv", index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(run_root) / "history.csv", index=False)

    if archive.values():
        pd.DataFrame(archive.values()).to_csv(Path(run_root) / "pareto_archive.csv", index=False)



if __name__ == "__main__":
    main()
