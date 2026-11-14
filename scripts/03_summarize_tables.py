"""All manuscript tables are produced here from predictions_all.csv (single source of truth)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C
from fluidml.metrics import summarize, score

tag = sys.argv[1] if len(sys.argv) > 1 else ""
pr = pd.read_csv(C.PREDS / f"predictions_all{tag}.csv")
T = C.TABLES
ORDER = {"xgb": 0, "rf": 1, "svr": 2, "pl": 3, "affine": 4}
def srt(d): return d.sort_values(["inputs", "model"], key=lambda s: s.map(ORDER) if s.name == "model" else s).reset_index(drop=True)

# T-A: row-level split, mean +- sd over seeds, and seed 0
rs = pr[pr.protocol == "row_split"]
per = summarize(rs, ["model", "inputs", "fold"])
agg = per.groupby(["model", "inputs"]).agg(**{f"{m}_{s}": (m, f) for m in ("R2_logZ", "RMSE_logZ", "R2_Z", "MAPE_Z", "R2_A", "FF2")
                                           for s, f in (("mean", "mean"), ("sd", "std"), ("min", "min"), ("max", "max"))}).reset_index()
agg.round(3).to_csv(T / "T_row_split_repeated.csv", index=False)
per[per.fold == "seed0"].round(3).to_csv(T / "T_row_split_seed0.csv", index=False)

# T-B: pooled out-of-fold performance for each grouped protocol
rows = []
for p in ("lobo_dev", "loso_dev", "transfer55", "lobo_transfer55"):
    rows.append(summarize(pr[pr.protocol == p], ["protocol", "model", "inputs"]))
tb = srt(pd.concat(rows)); tb.round(3).to_csv(T / "T_protocol_summary.csv", index=False)

# T-C: transfer55 by X-range subset and by soil (+ pressure_ahead horizon table)
tr = pr[pr.protocol == "transfer55"].copy()
tr["subset"] = np.where(tr.x_in_dev_range, "X within dev range (n=21)", "X above dev range (n=6)")
tc = pd.concat([summarize(tr, ["model", "inputs", "subset"]), summarize(tr, ["model", "inputs", "soil"]).rename(columns={"soil": "subset"})])
tc.round(3).to_csv(T / "T_transfer55_subsets.csv", index=False)
lt = pr[pr.protocol == "lobo_transfer55"].copy()
summarize(lt, ["model", "inputs", "soil"]).round(3).to_csv(T / "T_lobo_transfer55_by_soil.csv", index=False)
pa = pr[pr.protocol == "pressure_ahead"].copy()
pa["train_max_P"] = pa.fold.str.replace("trainmax", "").astype(int); pa["horizon_m"] = pa.P - pa.train_max_P
summarize(pa, ["model", "inputs", "train_max_P", "P"]).round(3).to_csv(T / "T_pressure_ahead.csv", index=False)

# T-D: per-bed errors (LOBO) and X vs sep paired comparison
pb = pr[pr.protocol.isin(["lobo_dev", "lobo_transfer55"])]
pbd = summarize(pb, ["protocol", "model", "inputs", "bed"])
pbd.round(3).to_csv(T / "T_per_bed.csv", index=False)
cmp = pbd.pivot_table(index=["protocol", "model", "bed"], columns="inputs", values="RMSE_logZ").reset_index()
cmp["sep_better"] = cmp["sep"] < cmp["X"]
cmp.groupby(["protocol", "model"]).agg(n_beds=("bed", "count"), beds_sep_better=("sep_better", "sum"),
                                      mean_RMSE_X=("X", "mean"), mean_RMSE_sep=("sep", "mean")).round(3).reset_index().to_csv(T / "T_inputs_paired_by_bed.csv", index=False)

# T-E: soil-specific bias at 55 m (mean signed log10 residual, Yhat - Y) for the transfer protocol
bias = tr.assign(res=tr.Y_pred - tr.Y_true).groupby(["model", "inputs", "soil"]).res.mean().unstack().round(3).reset_index()
bias.to_csv(T / "T_transfer55_soil_bias.csv", index=False)

# consistency identity (guards against the Table-4 type inconsistency): NRMSE^2 == (1-R2) * Var / range^2 for every row
chk = []
for key, g in pr.groupby(["protocol", "fold", "model", "inputs"]):
    Z, Zh = 10 ** g.Y_true.values, 10 ** g.Y_pred.values
    if len(Z) > 2 and Z.max() > Z.min():
        lhs = np.mean((Z - Zh) ** 2) / (Z.max() - Z.min()) ** 2
        rhs = (1 - (1 - ((Z - Zh) ** 2).sum() / ((Z - Z.mean()) ** 2).sum())) * Z.var() / (Z.max() - Z.min()) ** 2
        chk.append(abs(lhs - rhs))
print("max identity violation:", max(chk))
for f in ("T_protocol_summary", "T_row_split_repeated"):
    print(f); print(pd.read_csv(T / f"{f}.csv").to_string())
