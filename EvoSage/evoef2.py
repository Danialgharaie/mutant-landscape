import subprocess
import os
import shutil
from . import logger


def _ensure_b_factors(pdb_path: str) -> None:
  """Ensure B-factor values are present in ATOM/HETATM records.

  De-StReSS's PDB parser fails when the B-factor column is empty. EvoEF2
  outputs mutant structures without B-factors, so this helper fills missing
  values with ``0.00`` in-place.

  Parameters
  ----------
  pdb_path : str
      Path to the PDB file to modify.
  """
  try:
    with open(pdb_path, "r") as fh:
      lines = fh.readlines()
    new_lines = []
    for line in lines:
      if line.startswith(("ATOM", "HETATM")):
        if len(line) < 66:
          line = line.rstrip("\n").ljust(66) + "\n"
        if not line[60:66].strip():
          line = line[:60] + " 0.00" + line[66:]
      new_lines.append(line)
    with open(pdb_path, "w") as fh:
      fh.writelines(new_lines)
  except Exception as exc:
    logger.debug("Failed to patch B-factors for %s: %s", pdb_path, exc)


def build_mutant(pdb_fpath, mutant_file_path, output_dir, quiet: bool = True):
  """Builds mutant models using EvoEF2.

  Parameters
  ----------
  pdb_fpath : str
    Path to input PDB file.
  mutant_file_path : str
    Path to file with mutations (format: "CA171A,DB180E;").
  output_dir : str
    Output directory for mutant PDB.
  quiet : bool, optional
    If True (default), suppress EvoEF2 command logging and output.

  Returns
  -------
  str
    Path to generated mutant PDB file.

  Raises
  ------
  FileNotFoundError
    If required files are missing.
  RuntimeError
    If EvoEF2 execution fails.
  """
  # Setup paths and validate EvoEF2 executable
  evoef2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EvoEF2", "EvoEF2")
  evoef2_dir = os.path.dirname(evoef2_path)

  if not os.path.exists(evoef2_path):
    raise FileNotFoundError(f"EvoEF2 executable not found: {evoef2_path}")

  os.makedirs(output_dir, exist_ok=True)

  # Copy input files to EvoEF2's directory for local processing
  # Generate unique temporary names for copied files to avoid conflicts
  temp_pdb_name = "input_pdb_" + os.path.basename(pdb_fpath)
  temp_mut_name = "mut_file_" + os.path.basename(mutant_file_path)
  temp_pdb_path_in_evoef2_dir = os.path.join(evoef2_dir, temp_pdb_name)
  temp_mut_path_in_evoef2_dir = os.path.join(evoef2_dir, temp_mut_name)

  shutil.copy(pdb_fpath, temp_pdb_path_in_evoef2_dir)
  shutil.copy(mutant_file_path, temp_mut_path_in_evoef2_dir)

  # Get list of PDBs in evoef2_dir before running EvoEF2
  initial_pdbs_in_evoef2_dir = {f for f in os.listdir(evoef2_dir) if f.endswith(".pdb")}

  # Run EvoEF2 from its own directory with relative paths to copied inputs
  cmd = [
    "./" + os.path.basename(evoef2_path),
    "--command=BuildMutant",
    f"--pdb={temp_pdb_name}",
    f"--mutant_file={temp_mut_name}",
  ]
  if not quiet:
    logger.info("Executing EvoEF2 command: %s", ' '.join(cmd))
    logger.info("EvoEF2 working directory: %s", evoef2_dir)

  try:
    result = subprocess.run(cmd, cwd=evoef2_dir, check=True, capture_output=True, text=True)
    if not quiet:
      logger.info("EvoEF2 stdout:\n%s", result.stdout)
      if result.stderr:
        logger.info("EvoEF2 stderr:\n%s", result.stderr)

    # Get list of PDBs in evoef2_dir after running EvoEF2
    final_pdbs_in_evoef2_dir = {f for f in os.listdir(evoef2_dir) if f.endswith(".pdb")}

    # Find the new PDB file created by EvoEF2
    new_pdbs = list(final_pdbs_in_evoef2_dir - initial_pdbs_in_evoef2_dir)

    if not new_pdbs:
      raise RuntimeError(f"No new output PDB file found in {evoef2_dir} after EvoEF2 run. EvoEF2 stdout: {result.stdout}, stderr: {result.stderr}")
    
    # If multiple new PDBs are found (unlikely for single BuildMutant, but robust)
    # choose the most recently modified one.
    generated_pdb_filename = max(new_pdbs, key=lambda f: os.path.getctime(os.path.join(evoef2_dir, f)))
    generated_pdb_path_in_evoef2_dir = os.path.join(evoef2_dir, generated_pdb_filename)

    # Move the generated PDB to the intended output_dir
    final_pdb_path = os.path.join(output_dir, os.path.basename(generated_pdb_path_in_evoef2_dir))
    shutil.move(generated_pdb_path_in_evoef2_dir, final_pdb_path)

    _ensure_b_factors(final_pdb_path)

    return final_pdb_path

  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"EvoEF2 failed (code {e.returncode}):\n{e.stderr}")
  except Exception as e:
    raise RuntimeError(f"EvoEF2 error: {str(e)}")
  finally:
    # Clean up copied temporary files
    if os.path.exists(temp_pdb_path_in_evoef2_dir):
      os.remove(temp_pdb_path_in_evoef2_dir)
    if os.path.exists(temp_mut_path_in_evoef2_dir):
      os.remove(temp_mut_path_in_evoef2_dir)
