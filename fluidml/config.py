"""Central configuration. Every analysis choice that affects a reported number lives here."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("FLUID_ROOT", Path(__file__).resolve().parents[1]))
DATA_CSV = Path(os.environ.get("FLUID_DATA_CSV", ROOT / "data" / "raw" / "DataP.csv"))
RESULTS = ROOT / "results"
TABLES, FIGURES, PREDS, LOGS, CACHE = (RESULTS / s for s in
                                       ("tables", "figures", "predictions", "logs", "cache"))

# ---------------------------------------------------------------- units
# Confirmed by the authors: raw 'Af' is in cm^2.
# Unit of the raw 'Af' column. The original submission divided the raw numbers by
# d50 in metres squared, which is only dimensionally valid if Af is in m^2 (it is not:
# max Af would be ~118 m^2). cm^2 is the physically plausible unit for image-derived
# fluidized-zone areas of 1.9-120 units. R^2/NRMSE/MAPE of Z, and R^2 of log10(Z), are
# invariant to this choice (it only shifts log10(Z) by a constant), but dimensional-area
# results are reported in this unit.
AF_UNIT = os.environ.get("FLUID_AF_UNIT", "cm2")
AF_TO_MM2 = {"mm2": 1.0, "cm2": 1e2, "dm2": 1e4, "m2": 1e6}[AF_UNIT]

# Rows flagged as physically suspicious in the audit (kept in the primary analysis;
# a sensitivity run excludes them). (D50, CD, P, t) of the record.
SUSPECT_ROWS = [(0.64, 600, 15, 10)]   # Af = 32.28 while t=20,30 give 4.50, 5.39 (likely 3.228)

# ---------------------------------------------------------------- design
DEV_PRESSURES = (15, 25, 35, 45)       # development domain [m H2O]
TRANSFER_P = 55                        # withheld pressure [m H2O]
SEED = 42
LEGACY_SPLIT_SEED = 0                  # random_state used in the original submission
LEGACY_TEST_SIZE = 0.2
N_ROW_SPLIT_SEEDS = 10                 # repeated random row splits (seed 0 = original)

# ---------------------------------------------------------------- tuning
N_TRIALS = int(os.environ.get("FLUID_N_TRIALS", 30))
INNER_FOLDS = 5                        # GroupKFold over beds (grouped protocols)
N_JOBS = int(os.environ.get("FLUID_N_JOBS", 2))

# ---------------------------------------------------------------- bootstrap
N_BOOT = int(os.environ.get("FLUID_N_BOOT", 300))

# ---------------------------------------------------------------- engineering tolerance
# Prediction is 'acceptable' if the predicted area is within a factor of FACTOR of the
# measured one (0.5 <= A_hat/A <= 2). This is an author-chosen screening tolerance.
TOLERANCE_FACTOR = 2.0

MODELS_ML = ("xgb", "rf", "svr")
MODELS_PARAM = ("pl", "affine")        # power law (log-log OLS) and original Eq.(7)-type affine benchmark
INPUT_SETS = {
    "X":   ["X", "tau"],                         # composite descriptor (original)
    "sep": ["P_CD", "CD_d50", "tau"],            # separate dimensionless groups
}
