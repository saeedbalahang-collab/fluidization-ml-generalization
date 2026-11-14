"""Render manuscript tables (Markdown pipe tables) from result CSVs."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C, data as D
T = C.TABLES; M = T / "md"; M.mkdir(exist_ok=True)
NAME = {"xgb": "XGBoost", "rf": "RF", "svr": "SVR", "pl": "Power law", "affine": "Affine (Eq. 8)"}
INP = {"X": "Composite", "sep": "Separate"}
def md(df, path, align=None):
    cols = list(df.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    Path(path).write_text("\n".join(lines) + "\n")
f2 = lambda v: f"{v:.2f}"; f3 = lambda v: f"{v:.3f}"; pc = lambda v: f"{100*v:.0f}"
order = ["xgb", "rf", "svr", "pl", "affine"]
df = D.load(); leg = D.load(legacy=True)

# (Tables 2-4 are written by scripts/12_row_level_tables.py)

# T5 hyper-parameters
hp = pd.read_csv(T / "selected_hyperparameters.csv"); hp = hp[(hp.protocol == "transfer55") & hp.model.isin(["xgb", "rf", "svr"])]
rows = []
for r in hp.itertuples():
    p = json.loads(r.params); rows.append({"Model": NAME[r.model], "Inputs": INP[r.inputs], "Selected hyper-parameters": ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in p.items()), "Inner CV RMSE": f3(r.inner_cv_rmse)})
md(pd.DataFrame(rows), M / "T_hyperparameters.md")

# T6 within-domain
rs = pd.read_csv(T / "T_row_split_repeated.csv"); s0 = pd.read_csv(T / "T_row_split_seed0.csv"); ps = pd.read_csv(T / "T_protocol_summary.csv")
rows = []
for inp in ("X", "sep"):
    for m in order:
        if m == "affine" and inp == "sep": continue
        r = {"Inputs": INP[inp], "Model": NAME[m]}
        a = rs[(rs.model == m) & (rs.inputs == inp)]
        r["Row split, seed 0"] = f2(s0[(s0.model == m) & (s0.inputs == inp)].R2_logZ.iloc[0]) if len(a) else "–"
        r["Row split, mean ± SD"] = f"{a.R2_logZ_mean.iloc[0]:.2f} ± {a.R2_logZ_sd.iloc[0]:.2f}" if len(a) else "–"
        for p, lab in (("lobo_dev", "LOBO"), ("loso_dev", "LOSO")):
            g = ps[(ps.protocol == p) & (ps.model == m) & (ps.inputs == inp)].iloc[0]
            r[f"{lab} R²"] = f2(g.R2_logZ); r[f"{lab} FF2 (%)"] = pc(g.FF2)
        rows.append(r)
md(pd.DataFrame(rows), M / "T6.md")

# T7 55 m
rows = []
for inp in ("X", "sep"):
    for m in order:
        if m == "affine" and inp == "sep": continue
        g = ps[(ps.protocol == "transfer55") & (ps.model == m) & (ps.inputs == inp)].iloc[0]
        h = ps[(ps.protocol == "lobo_transfer55") & (ps.model == m) & (ps.inputs == inp)].iloc[0]
        rows.append({"Inputs": INP[inp], "Model": NAME[m], "R² (Y)": f2(g.R2_logZ), "RMSE (Y)": f2(g.RMSE_logZ),
                     "R² (A)": f2(g.R2_A), "MAPE (A, %)": f"{g.MAPE_A:.0f}", "FF2 (%)": pc(g.FF2),
                     "New bed: R² (Y)": f2(h.R2_logZ), "New bed: FF2 (%)": pc(h.FF2)})
md(pd.DataFrame(rows), M / "T7.md")

# T8 subsets
sub = pd.read_csv(T / "T_transfer55_subsets.csv"); rows = []
for m, inp in (("xgb", "X"), ("xgb", "sep"), ("svr", "sep"), ("pl", "X"), ("pl", "sep")):
    r = {"Model": NAME[m], "Inputs": INP[inp]}
    for s in ("X within dev range (n=21)", "X above dev range (n=6)", "fine", "medium", "coarse"):
        g = sub[(sub.model == m) & (sub.inputs == inp) & (sub.subset == s)].iloc[0]
        r[{"X within dev range (n=21)": "X in range (21): RMSE / FF2", "X above dev range (n=6)": "X above range (6): RMSE / FF2", "fine": "Fine (9): bias / FF2", "medium": "Medium (9): bias / FF2", "coarse": "Coarse (9): bias / FF2"}[s]] = (f"{g.RMSE_logZ:.2f} / {pc(g.FF2)}%" if "range" in s else f"{g.bias_logZ:+.2f} / {pc(g.FF2)}%")
    rows.append(r)
md(pd.DataFrame(rows), M / "T8.md")

# T9 recalibration
rc = pd.read_csv(T / "T_recalibration.csv"); rows = []
lab = {"none": "None", "shift": "Constant shift (1 par.)", "linear": "Linear recalibration (2 par.)", "diag": "log X correction (2 + 1 par.)"}
for inp in ("X", "sep"):
    for m in ("none", "shift", "linear", "diag"):
        r = {"Inputs (XGBoost)": INP[inp], "Correction": lab[m]}
        for sc, nm in (("forward_chain", "Forward chaining"), ("lobo_calib", "LOBO calibration"), ("in_sample", "In-sample")):
            g = rc[(rc.model == "xgb") & (rc.inputs == inp) & (rc.scheme == sc) & (rc.method == m)].iloc[0]
            r[f"{nm}: R² / RMSE / FF2"] = f"{g.R2_logZ:.2f} / {g.RMSE_logZ:.2f} / {pc(g.FF2)}%"
        rows.append(r)
md(pd.DataFrame(rows), M / "T9.md")
print("ok")
