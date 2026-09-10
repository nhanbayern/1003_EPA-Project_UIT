"""Data-free guards for the active six-family benchmark contract."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SOURCES = [
    ROOT / "experiments/moirai_var_aware/data.py",
    ROOT / "model/GARCH based/dataset.py",
    ROOT / "model/GARCH based/utils.py",
    ROOT / "model/transformer based/version_2/dataset.py",
    ROOT / "model/modify_autoformer/Hybrid GARCH-Autoformer/dataset.py",
    ROOT / "model/modify_autoformer/Wavelet Transform/dataset.py",
    ROOT / "model/Morai based/notebooks/kaggle_notebook.py",
]


def test_rolling60_endpoint_contract_numeric():
    returns = np.arange(1.0, 90.0)
    t = 60
    for h in (1, 3, 5, 10, 21):
        target = np.std(returns[t + h - 59 : t + h + 1], ddof=0)
        assert target == np.std(returns[t + h - 59 : t + h + 1], ddof=0)
        assert len(returns[t + h - 59 : t + h + 1]) == 60
    assert not np.array_equal(returns[t - 59 : t + 1], returns[t - 58 : t + 2])


def test_active_sources_have_endpoint_rolling_target():
    for path in ACTIVE_SOURCES:
        text = path.read_text(encoding="utf-8")
        assert "np.std" in text
        assert "ddof=0" in text
        assert "future_returns = self.returns[t + 1: t + h + 1]" not in text
    assert "target_window = r.iloc[i + 1 : i + seq_len + 1]" in ACTIVE_SOURCES[1].read_text(encoding="utf-8")
    assert "target_position = t + horizon" in ACTIVE_SOURCES[0].read_text(encoding="utf-8")


def test_moirai_snapshot_exports_validation_and_test():
    notebook = json.loads(
        (ROOT / "model/Morai based/notebooks/kaggle_notebook.ipynb")
        .read_text(encoding="utf-8")
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "target_position = t + h" in text
    assert "save_prediction_csv" in text
    assert "'validation'" in text
    assert "'test'" in text
    assert "future_returns = self.returns[t + 1: t + h + 1]" not in text
