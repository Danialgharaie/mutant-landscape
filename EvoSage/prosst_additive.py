import os

import matplotlib.pyplot as plt
import numpy as np
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

# Load models and tokenizers once at module level
_MODEL = None
_TOKENIZER = None
_PROCESSOR = None


def _load_models():
  """Load the ProSST model, tokenizer, and processor if not already loaded.

  Returns
  -------
  tuple
    A tuple containing (model, tokenizer, processor)
  """
  global _MODEL, _TOKENIZER, _PROCESSOR
  try:
    if _MODEL is None:
      _MODEL = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
      _MODEL.eval()  # Set model to evaluation mode
    if _TOKENIZER is None:
      _TOKENIZER = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
    if _PROCESSOR is None:
      _PROCESSOR = PdbQuantizer()
    return _MODEL, _TOKENIZER, _PROCESSOR
  except Exception as e:
    raise RuntimeError(f"Failed to load ProSST models: {str(e)}")


def run_prosst(input_seq, pdb_fpath):
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

  Returns
  -------
  pred_scores : pandas.DataFrame
    The pivoted DataFrame containing the mutation score matrix.

  Raises
  ------
  FileNotFoundError
    If the PDB file does not exist.
  RuntimeError
    If there's an error during model inference or processing.
  """
  if not os.path.exists(pdb_fpath):
    raise FileNotFoundError(f"PDB file not found: {pdb_fpath}")

  if not input_seq or not all(aa in "".join(SINGLE_LETTER_CODES) for aa in input_seq):
    raise ValueError(f"Invalid protein sequence: {input_seq}")

  # Load models only once
  model, tokenizer, processor = _load_models()

  # Process structure
  structure_sequence = processor(pdb_fpath)
  structure_key = os.path.basename(pdb_fpath)
  structure_sequence_offset = [i + 3 for i in structure_sequence["2048"][structure_key]["struct"]]

  # Tokenize input sequence
  tokenized_res = tokenizer([input_seq], return_tensors="pt")
  input_ids = tokenized_res["input_ids"]
  attention_mask = tokenized_res["attention_mask"]
  structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

  # Run model inference
  with torch.no_grad():
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      ss_input_ids=structure_input_ids,
    )

  # Compute log probabilities and remove the special tokens at the beginning and end
  logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()

  # Get vocabulary indices for amino acids
  vocab = tokenizer.get_vocab()

  # Fully vectorized computation of mutation scores
  seq_length = len(input_seq)

  # Get indices for all amino acids and wild-type residues
  aa_indices = torch.tensor([vocab[aa] for aa in SINGLE_LETTER_CODES])
  wt_indices = torch.tensor([vocab[aa] for aa in input_seq])

  # Create position arrays for vectorized operations
  positions = np.arange(seq_length) + 1  # 1-indexed positions

  # Extract logits for all positions and amino acids at once - fully vectorized
  # Get wild-type logits for each position using advanced indexing
  pos_indices = torch.arange(seq_length)
  wt_logits = logits[pos_indices, wt_indices]

  # Expand dimensions for broadcasting
  wt_logits_expanded = wt_logits.unsqueeze(1).expand(-1, len(SINGLE_LETTER_CODES))

  # Use broadcasting to get all combinations of positions and amino acids
  pos_indices_expanded = pos_indices.unsqueeze(1).expand(-1, len(SINGLE_LETTER_CODES))
  aa_indices_expanded = aa_indices.unsqueeze(0).expand(seq_length, -1)

  # Get all mutation logits at once
  all_mt_logits = logits[pos_indices_expanded, aa_indices_expanded]

  # Calculate scores using vectorized operations (follows the mathematical formulation in README)
  # Score(F) = log P(x_{p_i} = f_i | x, s) - log P(x_{p_i} = w_i | x, s)
  score_matrix = all_mt_logits - wt_logits_expanded

  # Convert to numpy for easier manipulation with pandas
  # Ensure the tensor is on CPU before converting to numpy. This avoids
  # device-related errors when the model is moved to GPU by the caller.
  score_matrix_np = score_matrix.cpu().numpy()

  # Create DataFrame directly from the score matrix
  # First create a multi-index DataFrame
  df_data = {}
  for i, aa in enumerate(SINGLE_LETTER_CODES):
    df_data[aa] = score_matrix_np[:, i].round(4)

  # Create the DataFrame with position as index
  pred_matrix = pd.DataFrame(df_data, index=positions)

  # Add wild-type column and set index name
  pred_matrix.insert(0, "wt", list(input_seq))
  pred_matrix.index.name = "index"

  # Reset index to make "index" a regular column
  pred_matrix = pred_matrix.reset_index()

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

