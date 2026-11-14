# Machine-learning generalization for leakage-induced soil fluidization

Code, data and manuscript materials for  
**"Assessing Machine Learning Generalization under Shifted Hydraulic Conditions for Leakage-Induced Soil Fluidization around Buried Pipelines"**  
S. Ghorbany, S. Balahang, M. Ghodsian – Tarbiat Modares University (submitted to *Journal of Hydroinformatics*, HYDRO-D-26-00379).

Everything reported in the paper (tables, figures, selected hyper-parameters, predictions) is produced by the scripts in this repository from the raw laboratory data.

## What the analysis does

| Step | Script | Output |
|---|---|---|
| 1 | `01_audit_data.py` | source-data audit: d50 = 2.1 mm vs 2.0 mm, area unit, non-monotonic records, reproduction of the original numbers |
| 2 | `02_run_protocols.py` | nested, bed-grouped validation for all protocols × models × input sets (cached per fold) |
| 3 | `03_summarize_tables.py` | all scores from one predictions table (`results/predictions/predictions_all.csv`) |
| 4 | `04_recalibration.py` | post-hoc correction: forward chaining, leave-one-bed-out calibration, in-sample |
| 5 | `05_bootstrap_shap_pdp.py` | bed-level bootstrap, SHAP (development domain), partial dependence |
| 6–8 | `06_make_figures.py`, `07_make_manuscript_tables.py`, `08_powerlaw_exponents.py` | figures, Markdown tables, power-law exponents with bed-bootstrap CIs |
| 9 | `09_compare_with_reference.py` | compares a fresh run with `reference_results/` |

**Design.** 135 records = 9 soil beds (3 soils × 3 cover depths) × 5 pressures (15–55 m) × 3 times. The 55 m records are never used for fitting, scaling or tuning. All validation keeps the records of a bed together (`GroupKFold`, leave-one-bed-out, leave-one-soil-out, rolling pressure origin). Models: XGBoost, Random Forest, SVR (Optuna/TPE, seeded) and log–log power-law / affine benchmarks, each with composite inputs (X, τ) and separate inputs (P/CD, CD/d50, τ).

## Data and units

`data/raw/DataP.csv` (never modified). Columns and units are in `data/DATA_DICTIONARY.md`. **Af is in cm²**, d50 in mm, CD in mm, P in m H2O. Configuration (units, seeds, trials, tolerance) is in `fluidml/config.py`.

## Run in Google Colab (recommended)

1. Put this folder in Google Drive: `MyDrive/Fluidization/fluidization-ml-generalization/`.
2. Open `notebooks/colab_run.ipynb` in Colab and run all cells (≈ 25–40 min on a free CPU runtime; each fold is cached in `results/cache/`, so a disconnect only costs the unfinished fold).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q tests
python scripts/run_all.py --n-trials 30
```

`requirements-lock.txt` lists the exact package versions of the reference run. Tree-ensemble results can differ slightly across versions/platforms; `09_compare_with_reference.py` reports the maximum difference against `reference_results/`.

## Repository layout

```
data/raw/DataP.csv          raw data (unchanged)        fluidml/        package (config, data, metrics, models, tuning, protocols, recalibration)
data/DATA_DICTIONARY.md     units and definitions       scripts/        numbered analysis steps + run_all.py + build_manuscript.py
results/tables|figures|predictions|logs                 tests/          unit tests (data, units, metric identities)
reference_results/          reference-run headline tables    notebooks/  Colab runner
manuscript/                 manuscript sources and response to reviewers
```

## Reproducing the manuscript tables

`scripts/11_extra_outputs.py` (residual structure, perturbation test, forward-chaining correction details) and `scripts/12_row_level_tables.py` (row-level reference split tables) feed the figures and tables; `scripts/07_make_manuscript_tables.py` writes the Markdown tables inserted in `manuscript/manuscript.md`; `scripts/build_manuscript.py` builds the Word file (requires `pandoc` and `python-docx`).

## Publishing this repository on GitHub

```bash
git init && git add . && git commit -m "Initial release: analysis code, data and manuscript materials"
git branch -M main
git remote add origin https://github.com/<ORG>/fluidization-ml-generalization.git
git push -u origin main
```
`results/cache/` is git-ignored (large, regenerable). After acceptance, create a GitHub release and archive it on Zenodo to obtain the DOI quoted in the Data Availability Statement.

## Citation / licence
See `CITATION.cff`. Code: MIT licence. The laboratory data are the authors' own.

## Reproducing the figures

All figures except Figure 1 (the original experimental schematic, `results/figures/fig1_schematic.png`) are drawn by one script from saved result files, so no model has to be re-trained:

```bash
pip install -r requirements.txt
python scripts/06_make_figures.py      # writes results/figures/fig2…fig8 as PNG (500 dpi) and vector PDF; `13_make_workflow_pptx.py` builds the editable PowerPoint version of Fig. 3
```

Fonts, sizes and resolution are set at the top of `scripts/06_make_figures.py` (`plt.rcParams`, `savefig.dpi`); each figure is a clearly delimited block labelled `# --- Fig n`. The inputs are `results/predictions/*.csv` and `results/tables/*.csv`. To regenerate those inputs from the raw data use `python scripts/run_all.py` (about 1–3 h on a laptop CPU; `--n-trials` lowers the cost), or open `notebooks/colab_run.ipynb` in Google Colab with the data in `Drive/Fluidization/`. The PDF versions are editable vector graphics for journal submission.
