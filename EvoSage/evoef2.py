import subprocess
import os


def build_mutant(pdb_fpath, mutant_file_path, output_dir):
  """Builds mutant models using EvoEF2.

  Parameters
  ----------
  pdb_fpath : str
    Path to input PDB file.
  mutant_file_path : str
    Path to file with mutations (format: "CA171A,DB180E;").
  output_dir : str
    Output directory for mutant PDB.

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
  # Setup paths and validate inputs
  evoef2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EvoEF2", "EvoEF2")
  for path, desc in [(evoef2_path, "EvoEF2 executable"), (pdb_fpath, "PDB file"), (mutant_file_path, "mutant file")]:
    if not os.path.exists(path):
      raise FileNotFoundError(f"{desc} not found: {path}")

  os.makedirs(output_dir, exist_ok=True)

  # Run EvoEF2
  cmd = [
    evoef2_path,
    "--command=BuildMutant",
    f"--pdb={os.path.abspath(pdb_fpath)}",
    f"--mutant_file={os.path.abspath(mutant_file_path)}",
  ]

  try:
    result = subprocess.run(cmd, cwd=output_dir, check=True, capture_output=True, text=True)
    print(f"EvoEF2 stdout:\n{result.stdout}")
    if result.stderr:
      print(f"EvoEF2 stderr:\n{result.stderr}")

    # Get the output file from EvoEF2's output
    output_files = [f for f in os.listdir(output_dir) if f.endswith(".pdb")]
    if not output_files:
      raise RuntimeError(f"No output PDB files found. EvoEF2 output: {result.stdout} {result.stderr}")

    return os.path.join(output_dir, output_files[0])

  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"EvoEF2 failed (code {e.returncode}):\n{e.stderr}")
  except Exception as e:
    raise RuntimeError(f"EvoEF2 error: {str(e)}")


def run_destress(pdb_fpath, output_dir=None):
  """Runs DeStReSS analysis on a protein structure.

  Parameters
  ----------
  pdb_fpath : str
    Path to input PDB file to analyze.
  output_dir : str, optional
    Directory to run DeStReSS in. If None, uses the same directory as the input PDB.

  Raises
  ------
  FileNotFoundError
    If required files are missing.
  RuntimeError
    If DeStReSS execution fails.
  """
  # Setup paths and validate inputs
  destress_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "de-stress")
  destress_script = os.path.join(destress_dir, "run_destress_headless.py")

  if not os.path.exists(destress_script):
    raise FileNotFoundError(f"DeStReSS script not found: {destress_script}")
  if not os.path.exists(pdb_fpath):
    raise FileNotFoundError(f"PDB file not found: {pdb_fpath}")

  # Set output directory
  if output_dir is None:
    output_dir = os.path.dirname(pdb_fpath)
  os.makedirs(output_dir, exist_ok=True)

  # Run DeStReSS
  cmd = ["python3", destress_script, "--i", os.path.abspath(pdb_fpath)]

  try:
    result = subprocess.run(cmd, cwd=output_dir, check=True, capture_output=True, text=True)
    print(f"DeStReSS stdout:\n{result.stdout}")
    if result.stderr:
      print(f"DeStReSS stderr:\n{result.stderr}")

  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"DeStReSS failed (code {e.returncode}):\n{e.stderr}")
  except Exception as e:
    raise RuntimeError(f"DeStReSS error: {str(e)}")
