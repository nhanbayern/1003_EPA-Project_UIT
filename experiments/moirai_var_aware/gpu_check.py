from __future__ import annotations

import modal


app = modal.App("modal-gpu-check")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.3.1")


@app.function(image=image, gpu="A10G", timeout=10 * 60)
def check_gpu() -> str:
    import subprocess

    import torch

    result = subprocess.run(
        ["nvidia-smi"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [
        "torch.cuda.is_available(): %s" % torch.cuda.is_available(),
        "torch.version.cuda: %s" % torch.version.cuda,
    ]
    if torch.cuda.is_available():
        lines.append("torch.cuda.get_device_name(0): %s" % torch.cuda.get_device_name(0))
    lines.append("")
    lines.append("nvidia-smi:")
    lines.append(result.stdout if result.stdout else result.stderr)
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(check_gpu.remote())

