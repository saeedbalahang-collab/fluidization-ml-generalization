"""Source-data audit: d50 value, units of X and Z, time/pressure monotonicity, reproduction of the original numbers."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C, data as D
from fluidml.metrics import score, r2, nrmse, mape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

C.TABLES.mkdir(parents=True, exist_ok=True)
leg, new = D.load(legacy=True), D.load()
out = {}
# 1. d50 / X check
rows = []
for nm, d in (("original (d50=2.0 mm)", leg), ("corrected (d50=2.1 mm)", new)):
    dv, tr = D.dev(d), D.transfer(d)
    rows.append(dict(processing=nm, X_min=d.X.min(), X_dev_max=dv.X.max(), X_55_min=tr.X.min(), X_55_max=tr.X.max(),
                     n55_in_range=int(tr.x_in_dev_range.sum()), n55_out_range=int((~tr.x_in_dev_range).sum()),
                     Y_dev_min=dv.Y.min(), Y_dev_max=dv.Y.max(), Y_55_min=tr.Y.min(), Y_55_max=tr.Y.max()))
pd.DataFrame(rows).round(4).to_csv(C.TABLES / "audit_d50_X_Y.csv", index=False)
# 2. implied area in original processing
ymax = leg.Y.max(); dmin = 0.64e-3
out["legacy_max_Y"] = float(ymax)
out["legacy_implied_max_area_m2_if_Af_in_m2"] = float(10 ** ymax * dmin ** 2)
out["max_Af_raw"] = float(new.Af.max()); out["Af_unit_used"] = C.AF_UNIT
out["max_area_m2_corrected"] = float(new.Af.max() * C.AF_TO_MM2 / 1e6)
# 3. monotonicity
bad = []
for k, s in new.groupby(["D50", "CD", "P"]):
    a = s.sort_values("t").Af.values
    if not (a[0] <= a[1] <= a[2]): bad.append(dict(kind="time", D50=k[0], CD=k[1], P=k[2], Af_10_20_30=list(a)))
for k, s in new.groupby(["D50", "CD", "t"]):
    a = s.sort_values("P").Af.values
    if np.any(np.diff(a) < 0): bad.append(dict(kind="pressure", D50=k[0], CD=k[1], t=k[2], Af_by_P=list(a)))
pd.DataFrame(bad).to_csv(C.TABLES / "audit_nonmonotonic_records.csv", index=False)
out["n_nonmonotonic"] = len(bad); out["n_beds"] = int(new.bed.nunique()); out["rows_per_bed"] = int(len(new) / new.bed.nunique())
# 4. reproduce the ORIGINAL numbers (original processing, original hyperparameters)
cols = ["X", "tau"]
dv, tr = D.dev(leg), D.transfer(leg)
Zh7 = lambda s: 9510.31 * s.X + 6.914 * s.tau + 9.768
rep = {"eq7_dev_R2_NRMSE_MAPE": [r2(dv.Z, Zh7(dv)), nrmse(dv.Z, Zh7(dv)), mape(dv.Z, Zh7(dv))],
       "eq7_55m_R2_NRMSE_MAPE": [r2(tr.Z, Zh7(tr)), nrmse(tr.Z, Zh7(tr)), mape(tr.Z, Zh7(tr))],
       "eq7_dev_logRMSE": float(np.sqrt(np.mean((dv.Y - np.log10(Zh7(dv))) ** 2)))}
p = dict(max_depth=19, learning_rate=0.102, subsample=0.797, colsample_bytree=0.811, reg_lambda=0.0013, reg_alpha=0.0263, n_estimators=274)
seeds = []
for s in range(10):
    a, b = train_test_split(dv.index, test_size=0.2, random_state=s)
    sc = MinMaxScaler().fit(dv.loc[a, cols])
    m = XGBRegressor(**p, random_state=0, n_jobs=1).fit(sc.transform(dv.loc[a, cols]), dv.loc[a, "Y"])
    ph = lambda d: 10 ** m.predict(sc.transform(d[cols]))
    seeds.append(dict(split_seed=s, test_R2_Z=r2(dv.loc[b].Z, ph(dv.loc[b])), test_MAPE_Z=mape(dv.loc[b].Z, ph(dv.loc[b])),
                      R2_55=r2(tr.Z, ph(tr)), NRMSE_55=nrmse(tr.Z, ph(tr)), MAPE_55=mape(tr.Z, ph(tr))))
pd.DataFrame(seeds).round(3).to_csv(C.TABLES / "audit_legacy_reproduction_xgb.csv", index=False)
out["legacy_reproduction"] = rep
# 5. internal consistency identity NRMSE^2 = (1-R2) Var/range^2 on the reported Table 4 values
t4 = {"train": {"XGB": (.947, .052), "RF": (.880, .079), "SVR": (.774, .109), "Benchmark": (.119, .219)},
      "test": {"XGB": (.958, .048), "RF": (.867, .085), "SVR": (.852, .090), "Benchmark": (.119, .220)}}
out["table4_implied_var_over_range2"] = {s: {m: round(n ** 2 / (1 - r), 4) for m, (r, n) in d.items()} for s, d in t4.items()}
(C.TABLES / "audit_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2)); print(pd.DataFrame(rows).round(3).T); print(pd.DataFrame(seeds).round(3)); print(pd.DataFrame(bad))
