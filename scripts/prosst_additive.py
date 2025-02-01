import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from prosst.structure.quantizer import PdbQuantizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

SINGLE_LETTER_CODES = [
  "G",
  "A",
  "V",
  "L",
  "I",
  "T",
  "S",
  "M",
  "C",
  "P",
  "F",
  "Y",
  "W",
  "H",
  "K",
  "R",
  "D",
  "E",
  "N",
  "Q",
]


def run_prosst(input_seq, pdb_fpath, output_fpath):
  """
  Computes the predicted scores of all possible single point mutations of a given protein sequence
  using the ProSST model and saves the results in a matrix format. The saved CSV file has the columns:
    - index: the mutation position (1-indexed)
    - wt: the wild-type amino acid at that position
    - one column per amino acid (from SINGLE_LETTER_CODES) with the predicted score.

  Parameters
  ----------
  input_seq : str
      The input protein sequence.
  pdb_fpath : str
      The path to the PDB file of the protein structure.
  output_fpath : str
      The path to which the output file should be written.

  Returns
  -------
  pred_scores : pandas.DataFrame
      The pivoted DataFrame containing the mutation score matrix.
  """
  deprot = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
  processor = PdbQuantizer()

  structure_sequence = processor(pdb_fpath)
  structure_sequence_offset = [i + 3 for i in structure_sequence]

  tokenized_res = tokenizer([input_seq], return_tensors="pt")
  input_ids = tokenized_res["input_ids"]
  attention_mask = tokenized_res["attention_mask"]
  structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

  # Create a list of mutation labels (e.g. "M1A", "M1C", etc.)
  mutants = [f"{wt}{idx + 1}{mt}" for idx, wt in enumerate(input_seq) for mt in SINGLE_LETTER_CODES]

  with torch.no_grad():
    outputs = deprot(
      input_ids=input_ids,
      attention_mask=attention_mask,
      ss_input_ids=structure_input_ids,
    )
  # Compute log probabilities and remove the special tokens at the beginning and end.
  logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()

  vocab = tokenizer.get_vocab()
  # Create a long-format DataFrame of mutation scores.
  pred_scores = pd.DataFrame(columns=["mutant", "score"])
  for mutant in mutants:
    wt, idx, mt = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
    pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
    pred_scores.loc[len(pred_scores)] = {
      "mutant": mutant,
      "score": round(pred.item(), 4),
    }

  # Extract the mutation position (1-indexed) and target amino acid from the mutant label.
  pred_scores["Position"] = pred_scores["mutant"].str[1:-1].astype(int)
  pred_scores["mt"] = pred_scores["mutant"].str[-1]

  # Pivot so that each row corresponds to a mutation position and each column to a target amino acid.
  pivot = pred_scores.pivot(index="Position", columns="mt", values="score")

  # Create a column for the wild-type amino acid using input_seq.
  pivot.insert(0, "wt", list(input_seq))
  pivot.index.name = "index"

  # Reset the index so that "index" becomes a column.
  pred_matrix = pivot.reset_index()

  # Optionally, reorder the amino acid columns based on SINGLE_LETTER_CODES.
  cols = ["index", "wt"] + [aa for aa in SINGLE_LETTER_CODES if aa in pred_matrix.columns]
  pred_matrix = pred_matrix[cols]

  # Save the pivoted matrix.
  pred_matrix.to_csv(f"{output_fpath}/pred_scores.csv", index=False)

  return pred_matrix


def plot_landscape(pred_scores, output_fpath):
  """
  Plots a 3D fitness landscape using the given predicted scores in long format.
  This function expects the original long-format DataFrame, so it extracts
  the necessary information from the 'mutant' column. Note that if you require
  a landscape plot, you might want to generate the long-format DF separately.

  Parameters
  ----------
  pred_scores : pandas.DataFrame
      A DataFrame containing the mutation scores. If using the pivoted matrix,
      the function will extract the long format from the 'index' and 'wt' columns.
  output_fpath : str
      The path to which the plot should be saved.

  Returns
  -------
  None
  """
  # To plot, we need a long-format DataFrame.
  # Recreate the long format from the pivoted matrix.
  long_format = pd.melt(pred_scores, id_vars=["index", "wt"], var_name="mutant", value_name="score")
  # Exclude rows where 'score' is NaN (if any)
  long_format = long_format.dropna(subset=["score"])
  # For plotting, assume the wild-type amino acid is not needed.
  long_format["Position"] = long_format["index"]
  long_format["Amino Acid"] = long_format["mutant"]
  long_format["AA_Code"] = long_format["Amino Acid"].astype("category").cat.codes

  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(long_format["Position"], long_format["AA_Code"], long_format["score"], c=long_format["score"], cmap="viridis", depthshade=True)
  ax.set_xlabel("Position")
  ax.set_ylabel("Amino Acid")
  ax.set_zlabel("Score")

  aa_categories = long_format["Amino Acid"].astype("category").cat.categories
  ax.set_yticks(range(len(aa_categories)))
  ax.set_yticklabels(aa_categories)

  plt.savefig(f"{output_fpath}/landscape.png")


def plot_heatmap(pred_scores, output_fpath):
  """
  Plots a heatmap of the mutation scores from the pivoted matrix.
  The x-axis corresponds to the mutated amino acid and the y-axis to the mutation position.

  Parameters
  ----------
  pred_scores : pandas.DataFrame
      The pivoted DataFrame containing the mutation score matrix.
  output_fpath : str
      The path to which the heatmap image should be saved.

  Returns
  -------
  None
  """
  # Create a copy and set the index to "index" (mutation position).
  heatmap_df = pred_scores.set_index("index")
  # Remove the 'wt' column, as it is not used in the heatmap.
  if "wt" in heatmap_df.columns:
    heatmap_df = heatmap_df.drop(columns=["wt"])

  plt.figure(figsize=(10, 8))
  sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis")
  plt.xlabel("Mutated Amino Acid")
  plt.ylabel("Position")
  plt.title("Mutation Score Heatmap")
  plt.tight_layout()
  plt.savefig(f"{output_fpath}/heatmap.png")


def main():
  """
  The main entry point of the script.

  This function parses command line arguments and orchestrates the workflow
  of the script. It takes three required arguments: input_seq, pdb_fpath,
  and output_fpath. The first argument corresponds to the input sequence
  for which the fitness landscape should be computed. The second argument
  corresponds to the path of the PDB file from which the structural
  information is extracted. The third argument corresponds to the path
  where the output files should be written.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_seq", type=str, required=True)
  parser.add_argument("--pdb_fpath", type=str, required=True)
  parser.add_argument("--output_fpath", type=str, required=True)
  args = parser.parse_args()

  os.makedirs(args.output_fpath, exist_ok=True)

  # Run the ProSST model and save the pivoted (matrix) DataFrame.
  pred_scores = run_prosst(args.input_seq, args.pdb_fpath, args.output_fpath)
  plot_landscape(pred_scores, args.output_fpath)
  plot_heatmap(pred_scores, args.output_fpath)


if __name__ == "__main__":
  main()
