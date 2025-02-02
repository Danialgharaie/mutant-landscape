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
  pred_matrix.to_csv(os.path.join(output_fpath, "pred_scores.csv"), index=False)

  return pred_matrix


def plot_heatmap(pred_scores, output_fpath):
  pivoted = pred_scores.set_index("index").drop(columns=["wt"], errors="ignore")
  num_rows, num_cols = pivoted.shape

  height = max(8, num_rows * 0.10)
  width = 4.8

  plt.figure(figsize=(width, height))
  ax = sns.heatmap(pivoted, cmap="Blues", cbar_kws={"label": "Score"}, yticklabels=True, annot=False)

  ax.tick_params(axis="y", labelsize=8)
  ax.set_xlabel("Amino Acid")
  ax.set_ylabel("Position")
  ax.set_title("Fitness Landscape Heatmap")

  plt.tight_layout()
  plt.savefig(os.path.join(output_fpath, "landscape_heatmap.png"), dpi=300)
  plt.close()


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
  plot_heatmap(pred_scores, args.output_fpath)


if __name__ == "__main__":
  main()
