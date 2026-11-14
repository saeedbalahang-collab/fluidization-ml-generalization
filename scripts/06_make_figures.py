"""All manuscript figures (PNG 500 dpi + vector PDF) from the saved result tables.
Figures are drawn at their final print width (190 mm, Elsevier double column) so that the font sizes below are the printed sizes.
Figure 1 is the original schematic (results/figures/fig1_schematic.png) and is not regenerated here."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C, data as D
from fluidml.metrics import r2

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                     "legend.fontsize": 8.5, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 500, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans", "lines.markeredgewidth": 0.4})
W = 7.48  # inches = 190 mm
COL = {"fine": "#0072B2", "medium": "#E69F00", "coarse": "#009E73"}
MK = {"fine": "o", "medium": "s", "coarse": "^"}
LAB = {"fine": "fine (0.64 mm)", "medium": "medium (1.26 mm)", "coarse": "coarse (2.1 mm)"}
F = C.FIGURES; F.mkdir(parents=True, exist_ok=True)
def save(fig, name):
    fig.savefig(F / f"{name}.png"); fig.savefig(F / f"{name}.pdf")
df = D.load(); pr = pd.read_csv(C.PREDS / "predictions_all.csv"); pr["tau"] = pr.t / 30
LOGZ = r"$\log_{10}(A_f/d_{50}^2)$"; L10 = r"log$_{10}$ units"; EDGE = dict(edgecolors="k", linewidths=0.3)
def sel(p, m="xgb", i="X"): return pr[(pr.protocol == p) & (pr.model == m) & (pr.inputs == i)]
def panel(ax, s): ax.text(-0.02, 1.05, s, transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")
def one2one(ax, lo, hi, factor=True):
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.2, label="1:1")
    if factor:
        d = np.log10(C.TOLERANCE_FACTOR)
        ax.plot([lo, hi], [lo + d, hi + d], "k--", lw=.8, label=f"factor of {C.TOLERANCE_FACTOR:g}")
        ax.plot([lo, hi], [lo - d, hi - d], "k--", lw=.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")

# ---------------- Fig 2
fig, ax = plt.subplots(1, 3, figsize=(W, 3.0))
for a, c, xl, lg in zip(ax, ["P_CD", "CD_d50", "X"], ["$P/CD$ (-)", "$CD/d_{50}$ (-)", r"$X=P/\sqrt{CD\,d_{50}}$ (-)"], [False, True, False]):
    for s in COL:
        d = df[df.soil == s]; a.scatter(d[c], d.Y, s=30, c=COL[s], marker=MK[s], alpha=.9, label=LAB[s], **EDGE)
    a.set_xlabel(xl); a.set_ylabel(LOGZ)
    if lg:
        a.set_xscale("log"); a.set_xticks([200, 300, 500, 800]); a.set_xticklabels(["200", "300", "500", "800"]); a.minorticks_off()
for a, s in zip(ax, "abc"): panel(a, f"({s})")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, markerscale=1.3)
fig.tight_layout(); save(fig, "fig2_inputs_vs_response"); plt.close()

# ---------------- Fig 3 workflow (static version; the editable version is built by scripts/13_make_workflow_pptx.py)
fig, ax = plt.subplots(figsize=(W, 4.9)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(-0.3, 6.4)
def box(x, y, w, h, t, fc="#EEF3F8", ec="#33506B", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#33506B", lw=1.2))
RF, RE = "#FBE9E7", "#A33A2A"
box(4.1, 5.5, 3.8, .8, "Experimental dataset: 135 records\n(9 beds × 5 pressures × 3 times)", "#FFF5E0", "#B07800")
box(0.2, 4.2, 4.4, .9, "Development domain: P = 15–45 m\n(108 records, 9 soil beds)")
box(7.4, 4.2, 4.4, .9, "Withheld domain: P = 55 m\n(27 records; never used for\nfitting, scaling or tuning)", RF, RE)
box(0.2, 3.0, 4.4, .9, "Inputs: X, τ  or  P/CD, CD/d₅₀, τ\nTarget: Y = log₁₀(A_f/d₅₀²)")
box(0.2, 1.7, 4.4, 1.0, "Grouped splits (all rows of a bed\nkept together): LOBO, LOSO,\nrolling pressure origin")
box(0.2, 0.2, 4.4, 1.2, "Training fold: scaler fitted on\ntraining data only → TPE\nhyper-parameter search\n(inner bed-grouped CV)")
box(4.95, 0.2, 2.1, 1.2, "Final model\nrefit on the\ntraining fold")
box(4.95, 1.7, 2.1, 1.0, "Held-out fold\n(scaler applied,\nnot refitted)")
box(4.95, 3.0, 2.1, .9, "Metrics: log-space,\narea, factor of 2")
box(7.4, 1.7, 4.4, 1.0, "55 m evaluation: model from\ndevelopment data → predict 55 m\n(overall / X in range / above / by soil)", RF, RE)
box(7.4, 0.2, 4.4, 1.2, "Forward-chaining correction test:\ntrain 15–35 m → calibrate 45 m\n→ test 55 m", RF, RE)
arr(5.2, 5.5, 2.4, 5.1); arr(6.8, 5.5, 9.6, 5.1); arr(2.4, 4.2, 2.4, 3.9); arr(2.4, 3.0, 2.4, 2.7); arr(2.4, 1.7, 2.4, 1.4)
arr(4.6, .8, 4.95, .8); arr(6.0, 1.4, 6.0, 1.7); arr(6.0, 2.7, 6.0, 3.0); arr(9.6, 4.2, 9.6, 2.7); arr(9.6, 1.7, 9.6, 1.4)
ax.text(6, -0.25, "TPE = tree-structured Parzen estimator; LOBO / LOSO = leave-one-bed / leave-one-soil-out", ha="center", fontsize=8.5, style="italic")
save(fig, "fig3_workflow"); plt.close()

# ---------------- Fig 4 within-domain (leave-one-bed-out predictions, four models)
fig, ax = plt.subplots(2, 2, figsize=(W, 7.6))
MC = {"xgb": ("XGBoost", "#D55E00", "o"), "rf": ("RF", "#009E73", "s"), "svr": ("SVR", "#0072B2", "^"), "affine": ("Regression benchmark", "#CC79A7", "D")}
lo, hi = df.Y.min() - .1, D.dev(df).Y.max() + .1
for m, (nm, c, mk) in MC.items():
    g = sel("lobo_dev", m)
    ax[0, 0].scatter(g.Y_true, g.Y_pred, s=34, c=c, marker=mk, alpha=.85, label=f"{nm} (R² = {r2(g.Y_true, g.Y_pred):.2f})", **EDGE)
    ax[0, 1].scatter(g.Y_pred, g.Y_true - g.Y_pred, s=30, c=c, marker=mk, alpha=.85, **EDGE)
one2one(ax[0, 0], lo, hi); ax[0, 0].set_xlabel("Measured " + LOGZ); ax[0, 0].set_ylabel("Predicted " + LOGZ)
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.99), frameon=False, markerscale=1.2)
ax[0, 1].axhline(0, c="k", lw=.9); ax[0, 1].set_xlabel("Predicted " + LOGZ); ax[0, 1].set_ylabel("Residual (measured − predicted)")
sv = np.load(C.PREDS / "shap_values_X.npy"); si = pd.read_csv(C.PREDS / "shap_inputs_X.csv"); rng = np.random.default_rng(0)
for j, c_ in enumerate(si.columns):
    v = si[c_].values; z = (v - v.min()) / (v.max() - v.min() + 1e-12)
    ax[1, 0].scatter(sv[:, j], np.full(len(v), 1 - j) + rng.uniform(-.22, .22, len(v)), c=z, cmap="viridis", vmin=0, vmax=1, s=22, edgecolors="none")
ax[1, 0].set_yticks([1, 0]); ax[1, 0].set_yticklabels(["$X$", r"$\tau$"], fontsize=11); ax[1, 0].set_ylim(-.6, 1.6); ax[1, 0].axvline(0, c="k", lw=.8)
ax[1, 0].set_xlabel("SHAP value (impact on predicted $Y$, " + L10 + ")")
sm = plt.cm.ScalarMappable(cmap="viridis"); cb = fig.colorbar(sm, ax=ax[1, 0], ticks=[0, 1], fraction=.05, pad=.02); cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Feature value", fontsize=9.5); cb.outline.set_linewidth(.6)
pt = pd.read_csv(C.TABLES / "T_perturbation.csv").set_index("model"); x0 = np.arange(4)
for k, (col, lab, c) in enumerate((("base", "unperturbed", "#332288"), ("X_pert", "$X$ ± 3 %", "#CC6677"), ("tau_pert", r"$\tau$ ± 5 %", "#44AA99"))):
    ax[1, 1].bar(x0 + (k - 1) * .27, [pt.loc[m, col] for m in ("xgb", "rf", "svr", "affine")], .27, color=c, label=lab, edgecolor="k", linewidth=.4)
ax[1, 1].set_xticks(x0); ax[1, 1].set_xticklabels(["XGBoost", "RF", "SVR", "Benchm."]); ax[1, 1].set_ylabel("R² of " + LOGZ); ax[1, 1].set_ylim(0, 1.0); ax[1, 1].legend(frameon=False, loc="upper right", ncol=1)
for a, s_ in zip(ax.ravel(), "abcd"): panel(a, f"({s_})")
fig.tight_layout(); save(fig, "fig4_within_domain"); plt.close()

# ---------------- Fig 5 transfer
t = sel("transfer55"); t = t.assign(res=t.Y_true - t.Y_pred)
TAUC = ((1 / 3, "#56B4E9"), (2 / 3, "#E69F00"), (1.0, "#882255"))
fig, ax = plt.subplots(2, 2, figsize=(W, 7.8))
for s in COL:
    g = t[t.soil == s]
    ax[0, 0].scatter(g.Y_true, g.Y_pred, s=40, c=COL[s], marker=MK[s], label=LAB[s], **EDGE)
    ax[0, 1].scatter(g.Y_pred, g.res, s=40, c=COL[s], marker=MK[s], **EDGE)
o = t[~t.x_in_dev_range]; ax[0, 0].scatter(o.Y_true, o.Y_pred, s=120, facecolors="none", edgecolors="k", lw=1.1, label="X above development range")
lo, hi = t.Y_true.min() - .3, t.Y_true.max() + .3; one2one(ax[0, 0], lo, hi)
ax[0, 0].set_xlabel("Measured " + LOGZ); ax[0, 0].set_ylabel("Predicted " + LOGZ)
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.99), frameon=False, markerscale=1.1)
ax[0, 1].axhline(0, c="k", lw=.9); ax[0, 1].set_xlabel("Predicted " + LOGZ); ax[0, 1].set_ylabel("Residual (measured − predicted)")
for tau, c in TAUC:
    g = t[np.isclose(t.tau, tau)]; ax[1, 0].scatter(g.X, g.res, s=40, c=c, label=rf"$\tau$ = {tau:.2f}", **EDGE)
ax[1, 0].axhline(0, c="k", lw=.9); xmx = D.dev(df).X.max(); ax[1, 0].axvline(xmx, c="k", ls=":", lw=1.1)
ax[1, 0].text(xmx + 30, -0.52, "development\n$X$ maximum", fontsize=8.5, va="bottom")
ax[1, 0].set_xlabel(r"$X=P/\sqrt{CD\,d_{50}}$ (-)"); ax[1, 0].set_ylabel("Residual (measured − predicted)"); ax[1, 0].legend(frameon=False, loc="upper left")
pdp = pd.read_csv(C.TABLES / "pdp_X.csv"); xm = D.dev(df).X
ax[1, 1].axvspan(xm.min(), xm.max(), color="#E8E8F2", label="development $X$ range"); ax[1, 1].plot(pdp.X, pdp.xgb, c="#D55E00", lw=1.6, label="XGBoost (partial dep.)")
ax[1, 1].plot(pdp.X, pdp.power_law, c="#0072B2", ls="--", lw=1.6, label="power law (log–log OLS)")
ax[1, 1].scatter(D.transfer(df).X, D.transfer(df).Y, s=16, c="k", label="55 m measurements"); ax[1, 1].set_xlabel(r"$X$ (-)"); ax[1, 1].set_ylabel(LOGZ + r" (avg. over $\tau$)"); ax[1, 1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=.8)
for a, s in zip(ax.ravel(), "abcd"): panel(a, f"({s})")
fig.tight_layout(); save(fig, "fig5_transfer_55m"); plt.close()

# ---------------- Fig 6 bootstrap
fig, ax = plt.subplots(1, 2, figsize=(W, 3.7)); b = pd.read_csv(C.PREDS / "bootstrap_X_bed.csv").sort_values(["soil", "X", "tau"]).reset_index(drop=True)
ax[0].fill_between(b.index, b.p05, b.p95, color="#9DC3E6", alpha=.7, label="5–95% bed-level bootstrap band"); ax[0].plot(b["mean"], c="#0072B2", lw=1.6, label="bootstrap mean"); ax[0].scatter(b.index, b.Y, s=30, c="k", label="measured", zorder=3)
ax[0].set_xlabel("55 m record (sorted by soil, $X$, $\\tau$)"); ax[0].set_ylabel(LOGZ); ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
ax[1].scatter(b.sd, (b.Y - b["mean"]).abs(), s=34, c="#0072B2", **EDGE); ax[1].set_xlabel("Bootstrap std. dev. of prediction (" + L10 + ")"); ax[1].set_ylabel("Absolute error (" + L10 + ")")
for a, s in zip(ax, "ab"): panel(a, f"({s})")
fig.tight_layout(); save(fig, "fig6_bootstrap"); plt.close()

# ---------------- Fig 7 input representation
pt = pd.read_csv(C.TABLES / "T_protocol_summary.csv"); fig, ax = plt.subplots(1, 3, figsize=(W, 3.1), sharey=True)
for a, (p, nm) in zip(ax, (("lobo_dev", "leave-one-bed-out\n(15–45 m)"), ("loso_dev", "leave-one-soil-out\n(15–45 m)"), ("transfer55", "withheld 55 m"))):
    for k, (i, col, lab) in enumerate((("X", "#0072B2", r"composite: $X$, $\tau$"), ("sep", "#E69F00", r"separate: $P/CD$, $CD/d_{50}$, $\tau$"))):
        v = [pt[(pt.protocol == p) & (pt.model == m) & (pt.inputs == i)].RMSE_logZ.iloc[0] for m in ("xgb", "rf", "svr", "pl")]
        a.bar(np.arange(4) + (k - .5) * .38, v, .38, color=col, label=lab, edgecolor="k", linewidth=.4)
    a.set_xticks(range(4)); a.set_xticklabels(["XGB", "RF", "SVR", "Power\nlaw"], fontsize=9); a.set_title(nm, fontsize=10)
ax[0].set_ylabel("RMSE of $Y$ (" + L10 + ")"); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
for a, s in zip(ax, "abc"): panel(a, f"({s})")
fig.tight_layout(); save(fig, "fig7_input_representation"); plt.close()

# ---------------- Fig 8 forward-chaining correction
fc = pd.read_csv(C.PREDS / "forward_chain_xgb_X.csv"); ts, cl = fc[fc.set == "test_55m"], fc[fc.set == "calibration_45m"]
fig, ax = plt.subplots(1, 3, figsize=(W, 4.0))
lo, hi = ts.Y_true.min() - .3, ts.Y_true.max() + .3
CB, CA = "#D55E00", "#009E73"   # before / after correction at 55 m
ax[0].scatter(ts.Y_true, ts.Y_pred, s=32, c=CB, marker="s", label=f"before (R² = {r2(ts.Y_true, ts.Y_pred):.2f})", **EDGE)
ax[0].scatter(ts.Y_true, ts.Y_corr, s=32, c=CA, marker="o", label=f"after (R² = {r2(ts.Y_true, ts.Y_corr):.2f})", **EDGE); one2one(ax[0], lo, hi)
ax[0].set_xlabel("Measured $Y$ at 55 m"); ax[0].set_ylabel("Predicted $Y$"); ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, handletextpad=.3)
for g, mk, nm, cb_, ca_ in ((cl, "o", "45 m", "#0072B2", "#CC79A7"), (ts, "s", "55 m", CB, CA)):
    ax[1].scatter(g.X, g.Y_true - g.Y_pred, s=30, c=cb_, marker=mk, label=f"before, {nm}", **EDGE)
    ax[1].scatter(g.X, g.Y_true - g.Y_corr, s=30, c=ca_, marker=mk, label=f"after, {nm}", **EDGE)
ax[1].axhline(0, c="k", lw=.9); ax[1].set_xlabel(r"$X$ (-)"); ax[1].set_ylabel("Residual of $Y$ (" + L10 + ")"); ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, columnspacing=.6, handletextpad=.2)
for tau, c in TAUC:
    g = ts[np.isclose(ts.tau, tau)].sort_values("X"); ax[2].scatter(g.X, g.Y_corr - g.Y_pred, s=32, c=c, label=rf"$\tau$ = {tau:.2f}", **EDGE)
ax[2].axhline(0, c="k", lw=.9); ax[2].set_xlabel(r"$X$ (-)"); ax[2].set_ylabel("Correction term (" + L10 + ")"); ax[2].legend(frameon=False, loc="lower right")
for a_, s_ in zip(ax, "abc"): panel(a_, f"({s_})")
fig.tight_layout(w_pad=1.2); save(fig, "fig8_correction"); plt.close()
print("figures:", sorted(p.name for p in F.glob("*.png")))
