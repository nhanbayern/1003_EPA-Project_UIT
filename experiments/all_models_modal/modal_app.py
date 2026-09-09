"""Unified Modal launcher for the project's existing benchmark notebooks.

Run from the repository root, for example:
  py -3.11 -m modal run experiments/all_models_modal/modal_app.py --families garch,transformer
"""
from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime
from pathlib import Path

import modal


_MODULE_PATH = Path(__file__).resolve()
# Locally the entrypoint is experiments/all_models_modal/modal_app.py; Modal
# imports a staged copy as /root/modal_app.py, where that parent hierarchy does
# not exist.  The source tree is explicitly mounted at /root/project below.
ROOT = _MODULE_PATH.parents[2] if len(_MODULE_PATH.parents) > 2 else Path("/root/project")
DATASET = ROOT / "dataset"
WEIGHTS = ROOT / "model" / "Morai based" / "weights"
APP_NAME = "volatility-benchmark-all-models"

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name("volatility-benchmark-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "arch", "torch==2.3.1", "pandas==2.1.4", "numpy==1.26.4",
        "scipy==1.11.4", "matplotlib", "nbformat", "rich", "ipython",
        "hydra-core", "jaxtyping", "jax[cpu]", "huggingface_hub",
        "safetensors", "einops", "gluonts", "ptwt", "PyWavelets",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/SalesforceAIResearch/uni2ts.git /root/uni2ts",
    )
    .add_local_dir(ROOT, remote_path="/root/project", ignore=[".git", ".venv", "output", "tmp"])
    .add_local_dir(DATASET, remote_path="/root/dataset")
    .add_local_dir(WEIGHTS, remote_path="/root/weights")
)

NOTEBOOKS = {
    "garch": "model/GARCH based/GARCH_Kaggle_Pipeline.ipynb",
    "transformer": "model/transformer based/version_2/kaggle_notebook_v2.ipynb",
    "moirai": "model/Morai based/notebooks/kaggle_notebook.ipynb",
    "moiraivar": "model/moirai_var_aware/moirai_kaggle.ipynb",
    "hybrid": "model/modify_autoformer/Hybrid GARCH-Autoformer/kaggle_notebook_hybrid.ipynb",
    "wavelet": "model/modify_autoformer/Wavelet Transform/kaggle_notebook_wavelet.ipynb",
}
STANDARD_COLUMNS = [
    "dataset", "branch", "tier", "model", "time", "horizon",
    "log_return", "true_volatility", "predict_volatility",
]


def _key(value: object) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _dataset_key(value: object) -> str:
    """Normalize legacy naming differences between notebooks and dataset files."""
    key = _key(value)
    return {"sp500": "snp500", "euronext100": "euronext100"}.get(key, key)


def _series_lookup(dataset_dir: Path) -> dict[str, tuple[list[object], list[float], dict[str, int]]]:
    """Return ordered trading calendars and one-step log returns for each dataset."""
    import numpy as np
    import pandas as pd

    lookups: dict[str, tuple[list[object], list[float], dict[str, int]]] = {}
    for csv_file in dataset_dir.glob("*.csv"):
        data = pd.read_csv(csv_file)
        data.columns = [column.strip().lower() for column in data.columns]
        if "close" not in data.columns:
            continue
        times = data.get("time", data.get("date", pd.Series(range(len(data)))))
        returns = np.log(pd.to_numeric(data["close"], errors="coerce") /
                         pd.to_numeric(data["close"], errors="coerce").shift(1)) * 100.0
        ordered_times = list(times)
        ordered_returns = [float(ret) if pd.notna(ret) else np.nan for ret in returns]
        lookups[_dataset_key(csv_file.stem)] = (
            ordered_times,
            ordered_returns,
            {_key(time): index for index, time in enumerate(ordered_times)},
        )
    return lookups


def _normalize_predictions(output_dir: Path) -> None:
    """Convert every baseline prediction artifact to the project-wide MoiraiVaR schema."""
    import numpy as np
    import pandas as pd

    destination = output_dir / "normalized_predictions"
    destination.mkdir(parents=True, exist_ok=True)
    series = _series_lookup(Path("/root/dataset"))

    def origin_and_next_return(dataset: str, target_time: object, horizon: int) -> tuple[object, float]:
        """Convert a horizon target timestamp back to its forecast origin on trading days."""
        calendar, log_returns, positions = series.get(_dataset_key(dataset), ([], [], {}))
        target_position = positions.get(_key(target_time))
        if target_position is None:
            return target_time, np.nan
        origin_position = target_position - int(horizon) + 1
        if origin_position < 0:
            return target_time, np.nan
        # MoiraiVaR convention: time=t and log_return=r_(t+1).
        next_position = origin_position + 1
        next_return = log_returns[next_position] if next_position < len(log_returns) else np.nan
        return calendar[origin_position], next_return

    for csv_file in (output_dir / "garch" / "predictions").glob("*_predictions.csv"):
        frame = pd.read_csv(csv_file)
        dataset = str(frame["dataset"].iloc[0])
        model = str(frame["model"].iloc[0])
        is_hybrid = model == "GARCH-LSTM-Hybrid"
        normalized = pd.DataFrame({
            "dataset": dataset,
            "branch": "GARCH_LSTM" if is_hybrid else "GARCH",
            "tier": "hybrid" if is_hybrid else "statistical",
            "model": model,
            "time": frame["time"],
            "horizon": frame["horizon"],
            # New GARCH exports already carry canonical origin-aligned
            # r[t+1].  Retain a fallback only for legacy artifacts.
            "log_return": (
                frame["log_return"]
                if "log_return" in frame.columns
                else [origin_and_next_return(dataset, time, 1)[1] for time in frame["time"]]
            ),
            "true_volatility": frame["actual_vol"],
            "predict_volatility": frame["pred_vol"],
        })
        normalized.to_csv(destination / f"{dataset}_{model}_predictions.csv", index=False)

    transformer_root = output_dir / "transformer" / "all_predictions"
    for csv_file in transformer_root.rglob("*_predictions.csv"):
        frame = pd.read_csv(csv_file)
        tier = csv_file.parent.name
        stem = csv_file.stem.removesuffix("_predictions")
        dataset, model = stem.rsplit("_", 1)
        normalized = pd.DataFrame({
            "dataset": dataset,
            "branch": "Transformers",
            "tier": tier,
            "model": model,
            # Transformer exports use the common origin t and r[t+1]
            # directly; do not reinterpret `time` as a target timestamp.
            "time": frame["time"],
            "horizon": frame["horizon"],
            "log_return": frame["log_return"],
            "true_volatility": frame["true_volatility"],
            "predict_volatility": frame["predict_volatility"],
        })
        normalized.to_csv(destination / f"{tier}_{dataset}_{model}_predictions.csv", index=False)

    for family_dir, branch in (("hybrid", "modified_autoformer"), ("wavelet", "modified_autoformer")):
        for csv_file in (output_dir / family_dir / "all_predictions").rglob("*_predictions.csv"):
            frame = pd.read_csv(csv_file)
            tier = csv_file.parent.name
            dataset, model = csv_file.stem.removesuffix("_predictions").rsplit("_", 1)
            normalized = pd.DataFrame({
                "dataset": dataset,
                "branch": branch,
                "tier": tier,
                "model": model,
                "time": frame["time"],
                "horizon": frame["horizon"],
                "log_return": frame["log_return"],
                "true_volatility": frame["true_volatility"],
                "predict_volatility": frame["predict_volatility"],
            })
            normalized.to_csv(destination / f"{family_dir}_{tier}_{dataset}_{model}_predictions.csv", index=False)

    for csv_file in (output_dir / "moirai" / "predictions").glob("*_predictions.csv"):
        frame = pd.read_csv(csv_file)
        dataset, model = csv_file.stem.removesuffix("_predictions").rsplit("_", 1)
        normalized = pd.DataFrame({
            "dataset": dataset,
            "branch": "Moirai",
            "tier": "baseline",
            "model": model,
            "time": frame["time"],
            "horizon": frame["horizon"],
            "log_return": frame["log_return"],
            "true_volatility": frame["true_volatility"],
            "predict_volatility": frame["predict_volatility"],
        })
        normalized.to_csv(destination / f"Moirai_{dataset}_{model}_predictions.csv", index=False)


def _prepare_notebook(source: Path, destination: Path, output_dir: Path, smoke_test: bool = False) -> None:
    """Make the committed Kaggle notebook portable without altering its source file."""
    import nbformat

    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        # Kaggle shell setup clones an old branch / placeholder URL; Modal already has source.
        lines = [line for line in cell.source.splitlines() if not line.lstrip().startswith("!")]
        text = "\n".join(lines)
        text = text.replace("/kaggle/working/1003_EPA-Project_UIT", "/root/project")
        text = text.replace("/kaggle/working/repo", "/root/project")
        text = text.replace("/kaggle/working/uni2ts/src", "/root/uni2ts/src")
        text = text.replace("'uni2ts/src/uni2ts/model'", "'/root/uni2ts/src/uni2ts/model'")
        text = text.replace("'uni2ts/src'", "'/root/uni2ts/src'")
        text = text.replace("/kaggle/input/datasets/trnhngv/historical-price", "/root/dataset")
        text = text.replace("/kaggle/input/datasets/trnhngv/weights", "/root/weights")
        text = text.replace("/kaggle/working/results_hybrid", str(output_dir / "hybrid"))
        text = text.replace("/kaggle/working/results_wavelet", str(output_dir / "wavelet"))
        text = text.replace("/kaggle/working/results_v2", str(output_dir / "transformer"))
        text = text.replace("/kaggle/working/results", str(output_dir / "garch"))
        text = text.replace("/kaggle/working/moirai_var", str(output_dir / "moiraivar" / "moirai_var"))
        text = text.replace("/kaggle/working/predictions", str(output_dir / "moirai" / "predictions"))
        text = text.replace("/kaggle/input/volatility-dataset/dataset/", "/root/dataset")
        text = text.replace("/kaggle/input/datasets/trnhngv/weights/weights", "/root/weights")
        # The baseline notebook's Kaggle path contains a duplicated `weights`
        # component; the Modal mount is already the weights directory itself.
        text = text.replace("/root/weights/weights", "/root/weights")
        if smoke_test:
            text = text.replace("for index_name in SPLIT_INFO.keys():", "for index_name in list(SPLIT_INFO.keys())[:1]:")
            # Some statistical notebooks enumerate raw dataset files instead
            # of SPLIT_INFO; keep smoke runs to one market in that path too.
            text = text.replace("for csv_file in csv_files:", "for csv_file in csv_files[:1]:")
            text = text.replace("EPOCHS = 30", "EPOCHS = 1")
            text = text.replace("epochs=30", "epochs=1")
            text = text.replace("for epoch in range(EPOCHS):", "for epoch in range(min(EPOCHS, 1)):")
        # Preserve trained MoiraiVaR state dicts, which the original notebook only exported as CSV.
        marker = "csv_outputs[filename] = frame.to_csv(index=False)"
        if marker in text:
            text = text.replace(
                marker,
                marker + "\n            weights_path = f'/root/modal_output/moiraivar/weights/{index_name}_{model_type}{mode_suffix}_lambda_{lambda_var:g}.pt'\n            os.makedirs(os.path.dirname(weights_path), exist_ok=True)\n            torch.save(model.state_dict(), weights_path)",
            )
        baseline_marker = "df_out.to_csv(csv_filename, index=False)"
        if baseline_marker in text:
            text = text.replace(
                baseline_marker,
                baseline_marker + "\n        weights_path = os.path.join('/root/modal_output/moirai/weights', f'{index_name}_{m_type}.pt')\n        os.makedirs(os.path.dirname(weights_path), exist_ok=True)\n        torch.save(model.state_dict(), weights_path)",
            )
        # Emit structured milestones during the long-running legacy notebooks.
        text = text.replace(
            "for csv_file in csv_files:",
            "for csv_file in csv_files:\n        reporter.dataset(Path(csv_file).stem)",
        )
        text = text.replace(
            "for stat_model_name in MODEL_SPECS.keys():",
            "for stat_model_name in MODEL_SPECS.keys():\n            reporter.model(stat_model_name, tier='statistical')",
        )
        text = text.replace(
            "print(f\"--- Running GARCH-LSTM Hybrid ---\")",
            "reporter.model('GARCH-LSTM-Hybrid', tier='hybrid')\n        print(f\"--- Running GARCH-LSTM Hybrid ---\")",
        )
        text = text.replace(
            "for tier_name, config in TIERS_CONFIG.items():",
            "for tier_name, config in TIERS_CONFIG.items():\n    reporter.tier(tier_name)",
        )
        text = text.replace(
            "for m_name, model in models_dict.items():",
            "for m_name, model in models_dict.items():\n            reporter.model(m_name, tier=tier_name)",
        )
        text = text.replace(
            "for m_type in ['moirai', 'moirai2', 'moirai_moe']:",
            "for m_type in ['moirai', 'moirai2', 'moirai_moe']:\n        reporter.model(m_type, tier='baseline')",
        )
        text = text.replace(
            "for epoch in range(EPOCHS):",
            "for epoch in range(EPOCHS):\n        reporter.epoch(epoch + 1, EPOCHS)",
        )
        text = text.replace(
            "for epoch in range(epochs):",
            "for epoch in range(epochs):\n        reporter.epoch(epoch + 1, epochs)",
        )
        cell.source = text
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 8, volumes={"/root/persist": artifact_volume})
def run_selected(families: list[str], smoke_test: bool = False, run_id: str = "latest") -> bytes:
    import os
    import sys
    import time
    import nbformat
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    class Reporter:
        def __init__(self) -> None:
            self.console = Console(force_terminal=True, color_system="truecolor")
            self.family = ""
            self.current_model = ""
            self.current_dataset = ""
            self.started = time.monotonic()

        def begin(self, family: str) -> None:
            self.family = family
            self.console.print(Panel.fit(f"[bold cyan]START[/] [white]{family.upper()}[/]", border_style="cyan"))

        def dataset(self, name: str) -> None:
            self.current_dataset = name
            self.console.print(f"[bold yellow]DATASET[/] [white]{name}[/]")

        def tier(self, name: str) -> None:
            self.console.print(f"[bold magenta]TIER[/] [white]{name}[/]")

        def model(self, name: str, tier: str = "") -> None:
            self.current_model = name
            suffix = f"  [dim]tier={tier}[/]" if tier else ""
            self.console.print(f"[bold green]MODEL[/] [white]{name}[/]{suffix}")

        def epoch(self, current: int, total: int) -> None:
            # Rich output is retained in Modal Logs, unlike notebook cell output.
            self.console.print(
                f"[cyan]EPOCH[/] [bold]{current:>2}/{total}[/]  "
                f"[dim]{self.current_dataset} · {self.current_model}[/]"
            )

        def cell(self, current: int, total: int) -> None:
            self.console.print(f"[dim]executing cell {current}/{total} ({self.family})[/]")

        def done(self, family: str) -> None:
            elapsed = time.monotonic() - self.started
            self.console.print(Rule(f"[bold green]DONE {family.upper()} · {elapsed / 60:.1f} min[/]"))

    output_dir = Path("/root/modal_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = Reporter()
    family_paths = {
        "garch": "/root/project/model/GARCH based",
        "transformer": "/root/project/model/transformer based/version_2",
        "moirai": "/root/project/model/Morai based/notebooks",
        "moiraivar": "/root/project",
        "hybrid": "/root/project/model/modify_autoformer/Hybrid GARCH-Autoformer",
        "wavelet": "/root/project/model/modify_autoformer/Wavelet Transform",
    }
    for family in families:
        reporter.begin(family)
        source = Path("/root/project") / NOTEBOOKS[family]
        prepared = Path("/root/prepared_notebooks") / f"{family}.ipynb"
        _prepare_notebook(source, prepared, output_dir, smoke_test=smoke_test)
        # The legacy notebooks use generic module names (dataset, models,
        # config, utils).  Remove the preceding family's modules before
        # executing the next notebook, otherwise Transformer imports GARCH's
        # dataset.py in the same long-lived Modal process.
        for module_name in ("dataset", "models", "config", "utils", "losses"):
            sys.modules.pop(module_name, None)
        family_path = family_paths[family]
        sys.path[:] = [path for path in sys.path if path not in family_paths.values()]
        sys.path.insert(0, family_path)

        notebook = nbformat.read(prepared, as_version=4)
        namespace = {"__name__": "__main__", "__file__": str(prepared), "reporter": reporter}
        os.environ["PYTHONPATH"] = "/root/project:/root/uni2ts/src"
        for cell_number, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            reporter.cell(cell_number, len(notebook.cells))
            exec(compile(cell.source, f"{prepared}:cell-{cell_number}", "exec"), namespace)
        reporter.done(family)
    _normalize_predictions(output_dir)

    # Persist large model weights outside the function return payload. Returning
    # all Moirai state dicts from one BytesIO can exceed container memory.
    import shutil
    persistent_run = Path("/root/persist") / run_id
    if persistent_run.exists():
        shutil.rmtree(persistent_run)
    shutil.copytree(output_dir, persistent_run)
    artifact_volume.commit()

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.rglob("*"):
            relative_parts = {part.lower() for part in file.relative_to(output_dir).parts}
            is_weight_artifact = bool(relative_parts & {"weights", "models_weights", "model_params"}) or file.suffix.lower() in {".pt", ".pth", ".safetensors"}
            if file.is_file() and not is_weight_artifact:
                zf.write(file, file.relative_to(output_dir))
    return archive.getvalue()


@app.local_entrypoint()
def main(families: str = "garch,transformer,moiraivar", output_dir: str = "output", smoke_test: bool = False):
    selected = [item.strip().lower() for item in families.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(NOTEBOOKS))
    if invalid:
        raise ValueError(f"Unknown family: {', '.join(invalid)}. Choose: {', '.join(NOTEBOOKS)}")
    if not selected:
        raise ValueError("Select at least one family")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_run_dir = ROOT / output_dir / timestamp
    local_run_dir.mkdir(parents=True, exist_ok=False)
    # Start each family concurrently in its own isolated container.  Calling
    # spawn first for every family is important: waiting immediately would
    # serialize the workloads again.  Each completed call is downloaded as
    # soon as it is collected, so a later failure cannot erase earlier output.
    calls = {}
    for family in selected:
        print(f"starting Modal family in parallel: {family}")
        calls[family] = run_selected.spawn([family], smoke_test=smoke_test, run_id=f"{timestamp}/{family}")

    weight_paths = {
        "garch": "garch/model_params",
        "transformer": "transformer/models_weights",
        "moirai": "moirai/weights",
        "moiraivar": "moiraivar/weights",
        "hybrid": "hybrid/models_weights",
        "wavelet": "wavelet/models_weights",
    }
    for family, call in calls.items():
        archive_path = local_run_dir / f"{family}_artifacts.zip"
        try:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(call.get())
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(local_run_dir)
            # Weights are persisted in the Modal Volume (not in the in-memory
            # return zip) and downloaded separately to avoid MemoryError.
            import subprocess
            local_weights = local_run_dir / family / Path(weight_paths[family]).relative_to(family)
            local_weights.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["py", "-3.11", "-m", "modal", "volume", "get",
                 "volatility-benchmark-artifacts", f"/{timestamp}/{weight_paths[family]}",
                 str(local_weights), "--force"],
                check=True,
            )
            print(f"downloaded completed family: {family} -> {local_run_dir}")
        except Exception as exc:
            print(f"family failed: {family}: {type(exc).__name__}: {exc}")
        finally:
            if archive_path.exists():
                archive_path.unlink()
    (local_run_dir / "run_manifest.txt").write_text(
        f"families={','.join(selected)}\nmodal_app={APP_NAME}\ncreated_at={timestamp}\n",
        encoding="utf-8",
    )
    print(f"saved artifacts to {local_run_dir}")
