#!/usr/bin/env python3
"""EcoSort - can TensorFlow train on an NVIDIA GPU on this machine?

    python ml/gpu_check.py            # report, always exits 0
    python ml/gpu_check.py --require  # exit 1 when no GPU is usable (what `make train-gpu` runs)

A GPU that TensorFlow silently fails to see does not produce an error - training just runs
on the CPU, 20-50x slower, and nobody notices until hours later. This prints what was found
and, when nothing was, which of the usual four causes it is.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def nvidia_smi() -> list[str] | None:
    """One 'name, driver, memory' line per card, or None when the driver is not installed."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def explain_no_gpu(tf, smi) -> None:
    system = platform.system()
    is_wsl = "microsoft" in platform.release().lower()
    build = tf.sysconfig.get_build_info()
    print("\nNo GPU is visible to TensorFlow. The likely cause:")
    if system == "Windows":
        print(
            "  * This is native Windows. TensorFlow dropped GPU support there after 2.10, so\n"
            "    train inside WSL2 (Ubuntu) instead - the Windows NVIDIA driver is shared with\n"
            "    WSL2 automatically. See ml/README.md, 'Training on an NVIDIA GPU'."
        )
    elif system == "Darwin":
        print("  * This is macOS: there is no CUDA on a Mac. Train on a Linux or WSL2 machine.")
    elif smi is None:
        where = "on Windows (not inside WSL2)" if is_wsl else "for this Linux system"
        print(
            f"  * `nvidia-smi` does not work, so the NVIDIA driver is missing or not loaded.\n"
            f"    Install the current NVIDIA driver {where}, reboot, and check that\n"
            "    `nvidia-smi` lists the card before trying again."
        )
    elif not build.get("is_cuda_build", False):
        print("  * This TensorFlow build has no CUDA support at all.")
    else:
        print(
            "  * The driver works but TensorFlow cannot load CUDA/cuDNN. Install the pip-packaged\n"
            "    CUDA libraries that match this TensorFlow:\n"
            "        make venv-gpu\n"
            "    (which runs: pip install 'tensorflow[and-cuda]==<the installed version>')"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report whether TensorFlow can train on an NVIDIA GPU here.")
    parser.add_argument("--require", action="store_true", help="Exit 1 when no usable GPU is found")
    parser.add_argument("--quiet", action="store_true", help="Only print problems")
    args = parser.parse_args(argv)
    say = (lambda *a, **k: None) if args.quiet else print

    try:
        import tensorflow as tf
    except ImportError as exc:
        print(f"error: TensorFlow is not installed ({exc}). Run: make venv")
        return 1 if args.require else 0

    smi = nvidia_smi()
    build = tf.sysconfig.get_build_info()
    say(f"TensorFlow   : {tf.__version__} (built for CUDA {build.get('cuda_version', 'none')}, "
        f"cuDNN {build.get('cudnn_version', 'none')})")
    say(f"Platform     : {platform.system()} {platform.release()}")
    if smi:
        for line in smi:
            say(f"nvidia-smi   : {line}")
    else:
        say("nvidia-smi   : not available")

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        explain_no_gpu(tf, smi)
        if args.require:
            print("\n`make train-gpu` needs a GPU; use `make train` to train on the CPU instead.")
            return 1
        return 0

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (ValueError, RuntimeError):
            pass

    tensor_cores = True
    for index, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        capability = details.get("compute_capability")
        name = details.get("device_name", gpu.name)
        say(f"GPU {index}        : {name}, compute capability "
            f"{'.'.join(map(str, capability)) if capability else 'unknown'}")
        if capability and tuple(capability) < (7, 0):
            tensor_cores = False

    # A real kernel launch, so a broken cuDNN/cuBLAS install fails here rather than
    # twenty minutes into the first epoch.
    try:
        with tf.device("/GPU:0"):
            a = tf.random.normal((2048, 2048))
            b = tf.random.normal((2048, 2048))
            (a @ b).numpy()
            start = time.perf_counter()
            for _ in range(10):
                c = a @ b
            c.numpy()
            elapsed = time.perf_counter() - start
        tflops = 10 * 2 * 2048 ** 3 / elapsed / 1e12
        say(f"Matmul test  : ok, ~{tflops:.1f} TFLOP/s float32")
    except Exception as exc:  # noqa: BLE001 - any failure here means "the GPU is not usable"
        print(f"error: a GPU is listed but running on it failed: {type(exc).__name__}: {exc}")
        return 1 if args.require else 0

    if tensor_cores:
        say("Mixed prec.  : supported - --mixed-precision (on in `make train-gpu`) roughly doubles speed")
    else:
        print(
            "note: this GPU predates tensor cores (compute capability < 7.0), so mixed precision\n"
            "      will not speed it up. Turn it off with:  make train-gpu GPU_PRECISION="
        )
    say("\nReady: `make train-gpu` will use the GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
