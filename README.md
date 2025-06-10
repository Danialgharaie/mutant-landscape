# **Protein Mutant Fitness Landscape Framework**

## **Overview**

This repository implements a **fitness landscape construction** method for protein mutations. The approach leverages **pre-trained protein language models (PLMs)** to systematically quantify mutation effects **without explicit supervision**.

The framework provides **additive scoring** for multi-site mutants and builds **per-position fitness landscapes** directly from language model probabilities. Epistatic effects are not explicitly modeled, but the search algorithms can explore multi-point combinations.



## **Features**

✔ **Fitness Landscape Construction**: Computes mutation scores across all possible amino acid substitutions for each position.  
✔ **Additive Scoring Model**: Evaluates mutations independently to approximate functional effects.  
✔ **Structure-Aware Predictions**: Uses **ProSST-derived structure tokens** as context.
✔ **Optional Bayesian Optimization**: Tune GA hyperparameters between runs.
✔ **Lightweight Surrogate Model**: Predict fitness values with caching.
✔ **Adaptive Mutation Rates**: Track successful mutations and update probabilities on the fly.
✔ **De-StReSS Evaluation**: Score structural stability via EvoEF2 and related tools.
✔ **Island Model Support**: Evolve multiple sub-populations with migration.
✔ **Local Optimization Modes**: Lamarckian or Baldwinian hill climbing steps.
✔ **Advanced Selection Operators**: Epsilon-dominant NSGA-II, MOEA/D, and hypervolume-based elitism.
✔ **Nondominated Archive and FASTA Output**: Persist Pareto-optimal sequences each generation.




## **Mathematical Formulation**


For a given **multi-point mutant** $F$, the **additive scoring function** is:

```math
Score(F) = \sum_{i=1}^{|F|} \left[ \log P(x_{p_i} = f_i | x, s) - \log P(x_{p_i} = w_i | x, s) \right]
```

Where:
- $p_i$ is the mutation position.
- $f_i$ is the mutated residue.
- $w_i$ is the original residue.
- $x$ is the wild-type amino acid sequence.
- $s$ is the **structural token sequence** from **ProSST**.

### **Fitness Landscape Calculation**
To construct a **per-position fitness landscape**, we iterate over all possible amino acid substitutions at each site:

1. Compute mutation effects **Δ** for all **20 amino acids** at each position:
   
```math   
Δ_k = \log P(x_{p} = AA_k | x, s) - \log P(x_{p} = w_p | x, s)
```   

2. Store results in a **fitness matrix** $Δ \in R^{L×20}$, where $L$ is the sequence length.


3. For multi-site mutations, sum the relevant **Δ** values.

⚠️ **Important Note**: This model assumes **independent additive contributions** and does not capture **epistatic interactions**.


## Using ProSST with your Python code

After running the setup script, if you want to use ProSST in your Python code, add it to your PYTHONPATH:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/ProSST"
```

You can add this line to your `.bashrc`, `.zshrc`, or run it in your shell before using the code.

## EvoEF2 and De-StReSS

`setup.sh` automatically clones the **EvoEF2** and **De-StReSS** repositories and builds the required binaries. EvoSage relies on these tools to generate mutant structures and compute stability metrics.


## Installing torch-scatter

`torch-scatter` must be installed using a version-specific wheel. For PyTorch 2.1.1 (CPU), run:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cpu.html
```

If you use a different PyTorch version, see: https://pytorch-geometric.com/whl/

## Running the Evolutionary Search

After running `setup.sh` to install dependencies and external tools, invoke the
genetic algorithm with:

```bash
python -m EvoSage.main "<WT_SEQUENCE>" path/to/structure.pdb --generations 50
```

Alternatively, provide the parameters in a JSON file and pass it via
`--config`. Every command-line option can be specified as a key. Any field
omitted from the JSON keeps the default value printed by `--help`. The keys
in this JSON mirror their corresponding CLI flags so new features like
`islands` or `bayes_opt` can also be configured here.

```json
{
  "wt_seq": "ACDE...",
  "pdb": "path/to/structure.pdb",
  "pop_size": 50,
  "max_k": 4,
  "generations": 20,
  "patience": 10,
  "beneficial_th": 0.5,
  "neutral_th": 0.0,
  "out_dir": "results",
  "dynamic_prosst": false,
  "mutation_prob": 0.08,
  "pm_start": null,
  "pm_min": null,
  "pm_decay": 1.0,
  "diversity_thresh": 0.0,
  "crossover_rate": 0.5,
  "cr_start": null,
  "cr_min": null,
  "cr_decay": 1.0,
  "log_level": "INFO",
  "seed": null,
  "islands": 1,
  "migration_interval": 5,
  "migrants": 2,
  "niche_clusters": 5,
  "niche_size": 3,
  "lamarck": false,
  "baldwin": false,
  "opt_steps": 5,
  "bayes_opt": false,
  "bayes_calls": 10,
  "use_surrogate": false,
  "surrogate_threshold": 0.0,
  "hv_selection": false,
  "moead_selection": false,
  "epsilon": 0.0,
  "adaptive_mutation": false
}
```

Run the search with:

```bash
python -m EvoSage.main --config config.json
```
Any CLI flags override values from the JSON file.

Use `--help` to see all available options. The script prints the final Pareto
front and always writes a `history.csv` log inside the run directory.

The `--log-level` flag controls logging verbosity and `--seed` sets the
Python and NumPy random seed for reproducible runs.

`--patience` specifies how many generations the algorithm tolerates without an
improved additive score. Once the limit is reached, the population is reset
around the best sequence found so far.

When `--dynamic_prosst` is enabled, EvoSage recomputes the ProSST score matrix
from the top sequence of each generation and updates the allowed mutation
dictionary accordingly before continuing.
`--mutation_prob` controls the per-site mutation probability used when
generating new candidates (default `0.08`). Higher values explore more mutations
each generation.
The adaptive schedule can be tuned with `--pm-start`, `--pm-min`,
`--pm-decay` and `--diversity-thresh`. These set the initial mutation rate,
minimum allowed rate, multiplicative decay factor and diversity threshold that
triggers decay. Crossover adapts using the analogous `--cr-start`, `--cr-min`
and `--cr-decay` options. Defaults keep both rates fixed when the parameters are
not provided.

Additional flags expose advanced behaviors:
* `--lamarckian` or `--baldwinian` perform a short local optimization step.
* `--islands` enables parallel island populations with periodic migration.
* `--hv-selection` and `--moead-selection` switch the selection strategy.
* `--bayes-opt` tunes hyperparameters, optionally assisted by `--use-surrogate`.
* `--adaptive-mutation` updates mutation probabilities from successful individuals.

## Plotting Results

The `EvoSage.plot_metrics` module provides helper functions to visualize the search progress. After running the evolutionary search, use the generated `history.csv` to create plots:

```python
from EvoSage.plot_metrics import plot_history, plot_final_scatter
plot_history("history.csv", "plots")
plot_final_scatter("history.csv", "plots")
```

`plot_history` shows how the average additive and z-score metrics evolve per generation, while `plot_final_scatter` plots a pairwise scatter matrix for the last generation.

## Advanced GA Options

When the search stagnates for `--patience` generations, EvoSage now rebuilds the
ProSST fitness matrix around the best sequence found so far. The allowed mutation
dictionary is recalculated from this new matrix and subsequent additive scoring
uses the updated values. This helps the algorithm escape local optima by
re-seeding the population with mutations that are neutral or beneficial relative
to the new best sequence.

Fallback mutations are guided by the ProSST matrix even when no allowed sites remain. Positions are ranked by the number and sum of negative scores, and random choices are drawn from the best-ranked (least deleterious) sites.

When running with multiple islands, each sub-population maintains an adaptive grid archive of elites. Individuals migrate between islands every few generations to share beneficial mutations.

## Self-Adaptive Parameters

Both the mutation probability and crossover rate can adapt during the search. At
each generation EvoSage adjusts these rates based on the population diversity and
whether the best additive score improved over the previous generation. Use
`--cr-start`, `--cr-min` and `--cr-decay` to control the adaptive schedule for
crossover in the same way that `--pm-start`, `--pm-min` and `--pm-decay` control
the mutation rate. The per-generation history of diversity, rates and best score
is saved to `stats_history.csv` for further analysis.
