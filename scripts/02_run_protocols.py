"""Run every (protocol, fold, model, input-set) combination; cache each result; write long predictions table.

usage: python scripts/02_run_protocols.py [--protocols a,b] [--n-trials N] [--drop-suspect]
"""
import argparse, json, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C, data as D, protocols as P
from fluidml.models import make_ml, make_param
from fluidml.tuning import tune


def run_one(df, job, model, iset, n_trials):
    cols = C.INPUT_SETS[iset]
    key = f"{job['protocol']}__{job['fold']}__{model}__{iset}__{n_trials}"
    cf = C.CACHE / (hashlib.md5(key.encode()).hexdigest()[:12] + ".json")
    if cf.exists():
        return json.loads(cf.read_text())
    tr, te = df.set_index("row_id").loc[job["train"]], df.set_index("row_id").loc[job["test"]]
    t0 = time.time()
    if model in C.MODELS_ML:
        params, cv_rmse, folds = tune(model, tr[cols], tr.Y, tr.bed, grouped=job["grouped_tuning"], n_trials=n_trials)
        m = make_ml(model, params).fit(tr[cols].values, tr.Y.values)
        info = dict(params=params, inner_cv_rmse=cv_rmse, inner_fold_rmse=folds)
    else:
        if model == "affine" and iset != "X":
            return None
        m = make_param(model).fit(tr[cols].values, tr.Y.values)
        info = dict(params=m.params)
    pred = m.predict(te[cols].values)
    res = dict(key=key, protocol=job["protocol"], fold=str(job["fold"]), model=model, inputs=iset,
               seconds=time.time() - t0, info=info, row_id=[int(i) for i in te.index], Y_pred=[float(v) for v in pred],
               n_train=len(tr))
    cf.write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default=",".join(P.ALL))
    ap.add_argument("--n-trials", type=int, default=C.N_TRIALS)
    ap.add_argument("--drop-suspect", action="store_true")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    for d in (C.CACHE, C.PREDS, C.TABLES, C.LOGS): d.mkdir(parents=True, exist_ok=True)
    df = D.load(drop_suspect=a.drop_suspect)
    if a.drop_suspect:
        df["row_id"] = np.arange(len(df))
    jobs = [(j, m, i) for p in a.protocols.split(",") for j in P.ALL[p](df)
            for m in (*C.MODELS_ML, *C.MODELS_PARAM) for i in C.INPUT_SETS
            if not (p == "row_split" and (i != "X" or m not in C.MODELS_ML))]
    print(len(jobs), "jobs", flush=True)
    # cache keys must differ for the sensitivity run
    if a.drop_suspect:
        for j, _, _ in jobs: j["fold"] = f"{j['fold']}|nosuspect"
    out = Parallel(n_jobs=C.N_JOBS, verbose=5)(delayed(run_one)(df, j, m, i, a.n_trials) for j, m, i in jobs)
    out = [o for o in out if o]
    base = df.set_index("row_id")
    rows, hp = [], []
    for o in out:
        sub = base.loc[o["row_id"]]
        rows.append(pd.DataFrame(dict(protocol=o["protocol"], fold=o["fold"].split("|")[0], model=o["model"], inputs=o["inputs"],
                                      row_id=o["row_id"], bed=sub.bed.values, soil=sub.soil.values, P=sub.P.values,
                                      CD=sub.CD.values, t=sub.t.values, X=sub.X.values, d50_mm=sub.d50_mm.values,
                                      x_in_dev_range=sub.x_in_dev_range.values,
                                      Y_true=sub.Y.values, Y_pred=o["Y_pred"])))
        hp.append(dict(protocol=o["protocol"], fold=o["fold"], model=o["model"], inputs=o["inputs"], n_train=o["n_train"],
                       seconds=round(o["seconds"], 1), **{k: json.dumps(v) for k, v in o["info"].items()}))
    suffix = "" if a.tag == "main" else "_" + a.tag
    pd.concat(rows).to_csv(C.PREDS / f"predictions_all{suffix}.csv", index=False)
    pd.DataFrame(hp).to_csv(C.TABLES / f"selected_hyperparameters{suffix}.csv", index=False)
    print("written", len(out), "results")


if __name__ == "__main__":
    main()
