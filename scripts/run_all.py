"""Run the complete analysis in order. usage: python scripts/run_all.py [--n-trials 30]"""
import subprocess, sys
from pathlib import Path
S = Path(__file__).parent; extra = sys.argv[1:]
steps = [("01_audit_data.py", []), ("02_run_protocols.py", extra), ("03_summarize_tables.py", []), ("04_recalibration.py", []),
         ("05_bootstrap_shap_pdp.py", []), ("08_powerlaw_exponents.py", []), ("11_extra_outputs.py", []), ("12_row_level_tables.py", []), ("06_make_figures.py", []), ("07_make_manuscript_tables.py", []),
         ("09_compare_with_reference.py", [])]
for f, a in steps:
    print(f"\n=== {f} ===", flush=True)
    subprocess.run([sys.executable, str(S / f), *a], check=True)
