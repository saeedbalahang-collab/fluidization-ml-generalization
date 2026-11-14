import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fluidml import data as D, metrics as M, config as C

def test_data_shape_and_beds():
    df = D.load(); assert len(df) == 135 and df.bed.nunique() == 9 and (df.groupby("bed").size() == 15).all()

def test_d50_is_2p1_and_X_range():
    df = D.load(); assert set(df.d50_mm.round(2)) == {0.64, 1.26, 2.1}
    assert abs(D.dev(df).X.max() - 2812.5) < 1e-6 and (D.transfer(df).x_in_dev_range.sum() == 21)

def test_legacy_reproduces_original_X_and_Y_ranges():
    lg = D.load(legacy=True); assert abs(lg.X.min() - 433.0127) < 1e-3 and abs(D.dev(lg).Y.min() - 5.6721) < 1e-3

def test_unit_conversion_is_constant_shift():
    # cm2 -> mm2 (x100) and d50 mm -> m in the legacy formula (1e6) => Y shifts by log10(100/1e6) = -4 for non-coarse soils
    df = D.load(); lg = D.load(legacy=True); m = df.D50 != 2.1
    assert np.allclose((df.Y - lg.Y)[m], -4.0, atol=1e-9)

def test_nrmse_r2_identity():
    rng = np.random.default_rng(0); y = rng.uniform(1, 9, 50); yh = y + rng.normal(0, .5, 50)
    assert abs(M.nrmse(y, yh) ** 2 - (1 - M.r2(y, yh)) * y.var() / (y.max() - y.min()) ** 2) < 1e-12

def test_ff2():
    assert M.within_factor([1, 1, 1], [0.6, 2.0, 2.1]) == 2 / 3
