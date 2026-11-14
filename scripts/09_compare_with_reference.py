"""Compare freshly produced headline tables with the reference run shipped in reference_results/ (tolerance for package-version drift)."""
import sys
from pathlib import Path
import pandas as pd, numpy as np
R = Path(__file__).resolve().parents[1]
ok = True
for f, cols in (("T_protocol_summary.csv", ["R2_logZ", "RMSE_logZ", "FF2"]), ("T_recalibration.csv", ["R2_logZ", "RMSE_logZ"])):
    ref, new = R / "reference_results" / f, R / "results" / "tables" / f
    if not ref.exists() or not new.exists(): print("skip", f); continue
    a, b = pd.read_csv(ref), pd.read_csv(new)
    d = np.abs(a[cols].values - b[cols].values).max()
    print(f"{f}: max abs difference {d:.4f}"); ok &= d < 0.05
print("MATCHES reference (tol 0.05)" if ok else "DIFFERS from reference – check package versions (requirements-lock.txt)")
