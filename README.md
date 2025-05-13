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
