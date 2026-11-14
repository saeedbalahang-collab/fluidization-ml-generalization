"""Power law with separate groups: coefficients, bed-level bootstrap CIs, and implied exponents of the dimensional area.
log10(Af/d50^2) = a + b1 log10(P/CD) + b2 log10(CD/d50) + b3 log10(tau)
=> Af ~ P^b1 * CD^(b2-b1) * d50^(2-b2) * tau^b3
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C, data as D
from sklearn.linear_model import LinearRegression
df = D.load(); cols = ["P_CD", "CD_d50", "tau"]
def fit(d):
    lr = LinearRegression().fit(np.log10(d[cols].values), d.Y.values); return np.r_[lr.intercept_, lr.coef_]
rng = np.random.default_rng(C.SEED); out = {}
for nm, d in (("development_15_45m", D.dev(df)), ("all_135", df)):
    est = fit(d); B = []
    for _ in range(2000):
        beds = rng.choice(d.bed.unique(), d.bed.nunique(), replace=True)
        B.append(fit(pd.concat([d[d.bed == b] for b in beds])))
    B = np.array(B)
    imp = lambda c: np.array([c[1], c[2] - c[1], 2 - c[2], c[3]])
    names = ["P", "CD", "d50", "tau"]
    out[nm] = dict(coef={"a": est[0], "b_P_CD": est[1], "b_CD_d50": est[2], "b_tau": est[3]},
                   ci95_bed_bootstrap={k: list(np.percentile(B[:, i], [2.5, 97.5])) for i, k in enumerate(["a", "b_P_CD", "b_CD_d50", "b_tau"])},
                   area_exponents={n: float(v) for n, v in zip(names, imp(est))},
                   area_exponents_ci95={n: list(np.percentile([imp(b)[i] for b in B], [2.5, 97.5])) for i, n in enumerate(names)})
(C.TABLES / "powerlaw_exponents.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out["development_15_45m"], indent=1, default=float))
