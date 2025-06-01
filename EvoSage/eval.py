import os
import subprocess

def run_destress(pdb_fpath, output_dir=None):
  """Runs DeStReSS analysis on a protein structure and parses the results.

  Parameters
  ----------
  pdb_fpath : str
    Path to input PDB file to analyze.
  output_dir : str, optional
    Directory to run DeStReSS in. If None, uses the same directory as the input PDB.

  Returns
  -------
  dict
    Dictionary containing parsed stability metrics including:
    - evoef2: total, ref_total, intraR_total, interS_total, interD_total
    - budeff: total, steric, desolvation, charge
    - rosetta: total and component scores
    - aggrescan3d: total_value, avg_value, min_value, max_value
    - Other metrics like hydrophobic_fitness, isoelectric_point, mass, etc.

  Raises
  ------
  FileNotFoundError
    If required files are missing.
  RuntimeError
    If DeStReSS execution fails or output parsing fails.
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

    # Parse the output into a dictionary of metrics
    metrics = {}
    
    # Extract EvoEF2 scores
    evoef2_scores = {}
    for line in result.stdout.split('\n'):
      if "EvoEF2 total energy:" in line:
        evoef2_scores['total'] = float(line.split(':')[1].strip())
      elif "EvoEF2 reference total energy:" in line:
        evoef2_scores['ref_total'] = float(line.split(':')[1].strip())
      elif "EvoEF2 intra-residue total energy:" in line:
        evoef2_scores['intraR_total'] = float(line.split(':')[1].strip())
      elif "EvoEF2 inter-residue total energy:" in line:
        evoef2_scores['interS_total'] = float(line.split(':')[1].strip())
      elif "EvoEF2 inter-domain total energy:" in line:
        evoef2_scores['interD_total'] = float(line.split(':')[1].strip())
    
    metrics['evoef2'] = evoef2_scores

    # Extract BUDE scores
    budeff_scores = {}
    for line in result.stdout.split('\n'):
      if "BUDE total energy:" in line:
        budeff_scores['total'] = float(line.split(':')[1].strip())
      elif "BUDE steric energy:" in line:
        budeff_scores['steric'] = float(line.split(':')[1].strip())
      elif "BUDE desolvation energy:" in line:
        budeff_scores['desolvation'] = float(line.split(':')[1].strip())
      elif "BUDE charge energy:" in line:
        budeff_scores['charge'] = float(line.split(':')[1].strip())
    
    metrics['budeff'] = budeff_scores

    # Extract Rosetta scores
    rosetta_scores = {}
    for line in result.stdout.split('\n'):
      if "Rosetta total score:" in line:
        rosetta_scores['total'] = float(line.split(':')[1].strip())
      elif "Rosetta fa_atr:" in line:
        rosetta_scores['fa_atr'] = float(line.split(':')[1].strip())
      elif "Rosetta fa_rep:" in line:
        rosetta_scores['fa_rep'] = float(line.split(':')[1].strip())
      elif "Rosetta fa_intra_rep:" in line:
        rosetta_scores['fa_intra_rep'] = float(line.split(':')[1].strip())
      elif "Rosetta fa_elec:" in line:
        rosetta_scores['fa_elec'] = float(line.split(':')[1].strip())
      elif "Rosetta fa_sol:" in line:
        rosetta_scores['fa_sol'] = float(line.split(':')[1].strip())
    
    metrics['rosetta'] = rosetta_scores

    # Extract Aggrescan3D scores
    aggrescan_scores = {}
    for line in result.stdout.split('\n'):
      if "Aggrescan3D total value:" in line:
        aggrescan_scores['total_value'] = float(line.split(':')[1].strip())
      elif "Aggrescan3D average value:" in line:
        aggrescan_scores['avg_value'] = float(line.split(':')[1].strip())
      elif "Aggrescan3D minimum value:" in line:
        aggrescan_scores['min_value'] = float(line.split(':')[1].strip())
      elif "Aggrescan3D maximum value:" in line:
        aggrescan_scores['max_value'] = float(line.split(':')[1].strip())
    
    metrics['aggrescan3d'] = aggrescan_scores

    # Extract other metrics
    for line in result.stdout.split('\n'):
      if "Hydrophobic fitness:" in line:
        metrics['hydrophobic_fitness'] = float(line.split(':')[1].strip())
      elif "Isoelectric point:" in line:
        metrics['isoelectric_point'] = float(line.split(':')[1].strip())
      elif "Molecular mass:" in line:
        metrics['mass'] = float(line.split(':')[1].strip().split()[0])  # Extract just the number
      elif "Number of residues:" in line:
        metrics['number_of_residues'] = int(line.split(':')[1].strip())
      elif "Packing density:" in line:
        metrics['packing_density'] = float(line.split(':')[1].strip())

    return metrics

  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"DeStReSS failed (code {e.returncode}):\n{e.stderr}")
  except Exception as e:
    raise RuntimeError(f"DeStReSS error: {str(e)}")
