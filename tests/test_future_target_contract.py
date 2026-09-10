"""Fast, data-free checks for the shared forecast-target contract."""

import importlib.util
import importlib
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aaai_windows_use_future_std_and_not_rolling_feature():
    module = _load_module(
        ROOT / "AAAI24_GARCH_NN_Reproduction" / "core" / "data_processor.py",
        "aaai24_data_processor",
    )
    returns = np.arange(1.0, 30.0)
    # Deliberately make the feature different from the target.
    historical_feature = np.full(returns.shape, 999.0)
    windows = module.create_sliding_windows(
        returns, historical_feature, seq_len=3, horizon=3, multi_horizon=False
    )

    np.testing.assert_allclose(windows["encoder_returns"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(windows["target_returns"][0], 6.0)
    np.testing.assert_allclose(windows["target_variance"][0], np.std([4.0, 5.0, 6.0], ddof=0))


def test_legacy_realized_vol_is_origin_aligned_future_std():
    # Avoid importing the legacy package __init__, which eagerly imports the
    # optional torch-based models.
    package = types.ModuleType("ultility")
    package.__path__ = [str(ROOT / "ultility")]
    sys.modules.setdefault("ultility", package)
    module = importlib.import_module("ultility.eval_vol")
    values = np.arange(1.0, 7.0)
    actual = module.realized_vol(values, horizon=2)
    expected = np.array(
        [np.std([2.0, 3.0], ddof=0), np.std([3.0, 4.0], ddof=0),
         np.std([4.0, 5.0], ddof=0), np.std([5.0, 6.0], ddof=0), np.nan, np.nan]
    )
    np.testing.assert_allclose(actual[:4], expected[:4])
    assert np.isnan(actual[4:]).all()
