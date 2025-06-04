import os
import subprocess
import csv
from . import logger

def run_destress(pdb_dir):
  """Runs DeStReSS analysis on all PDB files in a directory and parses the results from its CSV output.

  Parameters
  ----------
  pdb_dir : str
    Directory containing PDB files to analyze.

  Returns
  -------
  dict
    Dictionary containing the parsed metrics from the DeStReSS CSV output for all PDB files.
    Keys are PDB filenames, values are dictionaries of metrics.

  Raises
  ------
  FileNotFoundError
    If required files are missing.
  RuntimeError
    If DeStReSS execution fails or CSV parsing fails.
  """
  # Setup paths and validate inputs
  destress_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "de-stress")
  destress_script = os.path.join(destress_dir, "run_destress_headless.py")

  if not os.path.exists(destress_script):
    raise FileNotFoundError(f"DeStReSS script not found: {destress_script}")
  if not os.path.exists(pdb_dir):
    raise FileNotFoundError(f"Directory not found: {pdb_dir}")

  # Run DeStReSS on the directory
  cmd = ["python3", destress_script, "--i", os.path.abspath(pdb_dir)]
  logger.debug("Running De-StReSS: %s", ' '.join(cmd))

  try:
    result = subprocess.run(cmd, cwd=destress_dir, check=True, capture_output=True, text=True)
    if result.stderr:
      logger.debug("DeStReSS stderr:\n%s", result.stderr)

    # Find and parse the CSV file
    csv_files = [f for f in os.listdir(pdb_dir) if f.endswith('.csv')]
    if not csv_files:
      raise RuntimeError("No CSV output file found from DeStReSS")
    
    # Get the most recent CSV file
    latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(pdb_dir, x)))
    csv_path = os.path.join(pdb_dir, latest_csv)
    
    # Parse the CSV file
    with open(csv_path, 'r') as csvfile:
      reader = csv.DictReader(csvfile)
      # Convert all rows to a dictionary with design names as keys
      results = {}
      for row in reader:
        # Try several possible column names for the design name
        design_name = row.get('design name') or row.get('design_name') or row.get('file_name') or ''
        if design_name:
          results[design_name] = dict(row)
        else:
          logger.warning("No design name found in row. Keys: %s", list(row.keys()))

      if not results:
        raise RuntimeError("DeStReSS produced no results")

      ignore_keys = {'design name', 'design_name', 'file_name', 'file name'}
      for name, metrics in results.items():
        values = [v for k, v in metrics.items() if k not in ignore_keys]
        if not any(v not in (None, '', 'NA') for v in values):
          raise RuntimeError(
            f"DeStReSS returned empty metrics for {name}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
          )

      return results

  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"DeStReSS failed (code {e.returncode}):\n{e.stderr}")
  except Exception as e:
    raise RuntimeError(f"Error: {str(e)}")
