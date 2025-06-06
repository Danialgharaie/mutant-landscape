# **Protein Mutant Fitness Landscape Framework**

## **Overview**

This repository implements a **fitness landscape construction** method for protein mutations. The approach leverages **pre-trained protein language models (PLMs)** to systematically quantify mutation effects **without explicit supervision**.

The framework currently supports **additive scoring** for multi-site mutations and **per-position fitness landscapes**. Future updates will introduce **epistatic interaction analysis** for capturing higher-order mutation effects.



## **Features**

✔ **Fitness Landscape Construction**: Computes mutation scores across all possible amino acid substitutions for each position.  
✔ **Additive Scoring Model**: Evaluates mutations independently to approximate functional effects.  
✔ **Structure-Aware Predictions**: Uses **ProSST-derived structure tokens** as context.  

🚀 **Coming Soon**: Epistatic Interaction Analysis (Pairwise Effects)



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

⚠️ **Important Note**: This model assumes **independent additive contributions** and does not capture **epistatic interactions** (to be introduced in future updates).


## Using ProSST with your Python code

After running the setup script, if you want to use ProSST in your Python code, add it to your PYTHONPATH:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/ProSST"
```

You can add this line to your `.bashrc`, `.zshrc`, or run it in your shell before using the code.


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

Use `--help` to see all available options. The script prints the final Pareto
front and can also write a CSV log when `--output_csv` is provided.

The `--log-level` flag controls logging verbosity and `--seed` sets the
Python and NumPy random seed for reproducible runs.

`--patience` specifies how many generations the algorithm tolerates without an
improved additive score. Once the limit is reached, the population is reset
around the best sequence found so far.

When `--dynamic_prosst` is enabled, EvoSage recomputes the ProSST score matrix
from the top sequence of each generation and updates the allowed mutation
dictionary accordingly before continuing.

## Plotting Results

The `EvoSage.plot_metrics` module provides helper functions to visualize the search progress. After running the evolutionary search with `--output_csv run_history.csv`, generate plots with:

```python
from EvoSage.plot_metrics import plot_history, plot_final_scatter
plot_history("run_history.csv", "plots")
plot_final_scatter("run_history.csv", "plots")
```

`plot_history` shows how the average additive and z-score metrics evolve per generation, while `plot_final_scatter` plots a pairwise scatter matrix for the last generation.

## Advanced GA Options

When the search stagnates for `--patience` generations, EvoSage now rebuilds the
ProSST fitness matrix around the best sequence found so far. The allowed mutation
dictionary is recalculated from this new matrix and subsequent additive scoring
uses the updated values. This helps the algorithm escape local optima by
re-seeding the population with mutations that are neutral or beneficial relative
to the new best sequence.
