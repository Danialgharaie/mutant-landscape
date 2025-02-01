import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
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
  using the ProSST model.

  Parameters
  ----------
  input_seq : str
      The input protein sequence.
  pdb_fpath : str
      The path to the PDB file of the protein structure.
  output_fpath : str
      The path to which the output files should be written.

  Returns
  -------
  pred_scores : pandas.DataFrame
      A DataFrame containing the predicted scores of each mutation.
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

  mutants = [f"{wt}{idx + 1}{mt}" for idx, wt in enumerate(input_seq) for mt in SINGLE_LETTER_CODES]

  with torch.no_grad():
    outputs = deprot(
      input_ids=input_ids,
      attention_mask=attention_mask,
      ss_input_ids=structure_input_ids,
    )
  logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()

  vocab = tokenizer.get_vocab()
  pred_scores = pd.DataFrame(columns=["mutant", "score"])
  for mutant in mutants:
    wt, idx, mt = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
    pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
    pred_scores.loc[len(pred_scores)] = {
      "mutant": mutant,
      "score": round(pred.item(), 4),
    }

  pred_scores.to_csv(f"{output_fpath}/pred_scores.csv", index=False)

  return pred_scores


def plot_landscape(pred_scores, output_fpath):
  """
  Plots a 3D fitness landscape using the given predicted scores.

  This function takes in a DataFrame of predicted scores and an output path.
  It first reformats the DataFrame to be more suitable for plotting, and
  then uses matplotlib to create a 3D plot of the fitness landscape. The
  x-axis corresponds to the position of the mutation, the y-axis corresponds
  to the amino acid that is mutated, and the z-axis corresponds to the score
  of that mutation. The plot is then saved to the given output path.

  Parameters
  ----------
  pred_scores : pandas.DataFrame
      A DataFrame containing the predicted scores of each mutation.
  output_fpath : str
      The path to which the plot should be saved.

  Returns
  -------
  None
  """
  pred_scores["Position"] = pred_scores["mutant"].str[1:-1]
  pred_scores["Amino Acid"] = pred_scores["mutant"].str[0]
  pred_scores["Score"] = pred_scores["score"]

  pred_scores.plot(x="Position", y="Amino Acid", z="Score", kind="3d")
  plt.savefig(f"{output_fpath}/landscape.png")


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

  The function first creates the output directory if it does not exist.
  Then, it calls the run_prosst function with the parsed arguments and
  stores its output in the pred_scores variable. Finally, it calls the
  plot_landscape function with the pred_scores and output_fpath arguments.

  :return: None
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_seq", type=str, required=True)
  parser.add_argument("--pdb_fpath", type=str, required=True)
  parser.add_argument("--output_fpath", type=str, required=True)
  args = parser.parse_args()

  os.makedirs(args.output_fpath, exist_ok=True)

  pred_scores = run_prosst(args.input_seq, args.pdb_fpath, args.output_fpath)
  plot_landscape(pred_scores, args.output_fpath)
