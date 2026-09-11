from __future__ import annotations

from pathlib import Path

import modal


def _resolve_paths() -> tuple[Path, Path, Path, Path]:
    here = Path(__file__).resolve()

    if str(here) == "/root/modal_app.py" or Path("/root/moirai_var_aware").exists():
        remote_root = Path("/root")
        return (
            remote_root,
            remote_root / "moirai_var_aware",
            remote_root / "dataset",
            remote_root / "weights",
        )

    candidates = [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "experiments" / "moirai_var_aware").exists() and (candidate / "dataset").exists():
            return (
                candidate,
                candidate / "experiments" / "moirai_var_aware",
                candidate / "dataset",
                candidate / "model" / "Morai based" / "weights",
            )

    fallback = Path.cwd()
    return (
        fallback,
        fallback / "experiments" / "moirai_var_aware",
        fallback / "dataset",
        fallback / "model" / "Morai based" / "weights",
    )


PROJECT_ROOT, EXPERIMENT_DIR, DATASET_DIR, WEIGHTS_DIR = _resolve_paths()

app = modal.App("moirai-var-aware-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.1",
        "pandas==2.1.4",
        "numpy==1.26.4",
        "scipy==1.11.4",
        "hydra-core",
        "jaxtyping",
        "jax[cpu]",
        "huggingface_hub",
        "safetensors",
        "einops",
        "gluonts",
    )
    .run_commands("git clone --depth 1 https://github.com/SalesforceAIResearch/uni2ts.git /root/uni2ts")
    .add_local_dir(EXPERIMENT_DIR, remote_path="/root/moirai_var_aware")
    .add_local_dir(DATASET_DIR, remote_path="/root/dataset")
    .add_local_dir(WEIGHTS_DIR, remote_path="/root/weights")
)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 8)
def train_remote(
    lambda_var: float,
    epochs: int,
    batch_size: int,
    tuning_mode: str,
    backbone_lr: float,
    models: list[str],
    datasets: list[str] | None,
) -> bytes:
    import sys

    sys.path.insert(0, "/root")
    from moirai_var_aware.runner import run_training_to_zip

    return run_training_to_zip(
        lambda_var=lambda_var,
        epochs=epochs,
        batch_size=batch_size,
        tuning_mode=tuning_mode,
        backbone_lr=backbone_lr,
        models=models,
        datasets=datasets,
    )


@app.local_entrypoint()
def main(
    lambda_var: float = 0.2,
    epochs: int = 30,
    batch_size: int = 32,
    tuning_mode: str = "head",
    backbone_lr: float = 1e-5,
    models: str = "moirai,moirai2,moirai_moe",
    datasets: str = "",
    output_dir: str = "output/modal_moirai_var_loss",
    lambda_sweep: str = "",
):
    if tuning_mode not in {"head", "full"}:
        raise ValueError("tuning_mode must be 'head' or 'full'")
    selected_models = [m.strip() for m in models.split(",") if m.strip()]
    selected_datasets = [d.strip() for d in datasets.split(",") if d.strip()] or None
    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    mode_suffix = "" if tuning_mode == "head" else f"_{tuning_mode}"
    zip_name = f"moirai_var{mode_suffix}_lambda_{lambda_var:g}_predictions.zip"

    sweep = [float(value.strip()) for value in lambda_sweep.split(",") if value.strip()] if lambda_sweep else [lambda_var]
    for current_lambda in sweep:
        zip_bytes = train_remote.remote(current_lambda, epochs, batch_size, tuning_mode, backbone_lr, selected_models, selected_datasets)
        current_name = f"moirai_var{mode_suffix}_lambda_{current_lambda:g}_predictions.zip"
        out_file = output_path / current_name
        out_file.write_bytes(zip_bytes)
        print(f"saved {out_file}")
