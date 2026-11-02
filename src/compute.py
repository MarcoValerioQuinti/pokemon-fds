"""Portable hardware configuration (macOS / Linux / Windows).

Environment variables or script options --n-jobs / --device override all settings:
    POKEMON_N_JOBS   process/thread count (default: all cores)
    POKEMON_DEVICE   auto | cpu | cuda (default: auto -> cuda when an NVIDIA GPU is available)
    POKEMON_MP       fork | spawn (default: fork where available, spawn on Windows)

- Multiprocessing: fork on Linux/macOS (workers inherit parent data), spawn on Windows.
  Data is passed through the initializer, so both methods work.
- GPU: XGBoost (device="cuda") and CatBoost (task_type="GPU") support NVIDIA CUDA on
  Linux/Windows. Apple Silicon Macs use CPU; these models have no Apple GPU backend.
- LightGBM stays on CPU; the standard pip/conda build does not include GPU support.
"""
import multiprocessing as mp
import os
import shutil
import subprocess
import sys

_DEVICE = None


def n_jobs():
    return int(os.environ.get("POKEMON_N_JOBS", 0)) or os.cpu_count() or 1


def mp_context():
    forced = os.environ.get("POKEMON_MP")  # fork | spawn; use this to exercise Windows-style spawning elsewhere
    if forced:
        return mp.get_context(forced)
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods and sys.platform != "win32" else "spawn")


def _has_cuda():
    if sys.platform == "darwin" or not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def device():
    global _DEVICE
    if _DEVICE is None:
        want = os.environ.get("POKEMON_DEVICE", "auto")
        _DEVICE = ("cuda" if _has_cuda() else "cpu") if want == "auto" else want
    return _DEVICE


def configure(n=None, dev=None):
    """Apply command-line options before creating models or worker pools."""
    global _DEVICE
    if n:
        os.environ["POKEMON_N_JOBS"] = str(n)
    if dev:
        os.environ["POKEMON_DEVICE"] = dev
        _DEVICE = None


def xgb_params():
    return {"device": "cuda", "tree_method": "hist"} if device() == "cuda" else {"n_jobs": n_jobs()}


def catboost_params():
    return {"task_type": "GPU"} if device() == "cuda" else {"thread_count": n_jobs()}


def pool_map(func, items, initializer=None, initargs=(), chunksize=50):
    """Portable parallel map; a single job runs in the current process."""
    if n_jobs() == 1:
        if initializer:
            initializer(*initargs)
        return [func(x) for x in items]
    with mp_context().Pool(n_jobs(), initializer=initializer, initargs=initargs) as pool:
        return pool.map(func, items, chunksize=chunksize)


# Parallel folds per model family; threads per worker = n_jobs // workers.
# Measured on M5 (4 performance + 6 efficiency cores), 10k rows: boosting scales poorly beyond ~3 threads
# (lgbm/xgb: 3 workers x 3 threads are 1.5x faster than 1 x 10; catboost: 2 x 5 is 1.1x faster);
# sklearn ExtraTrees/RF scale well across threads and use one worker with all threads.
FOLD_WORKERS = {"lgbm": 3, "xgb": 3, "catboost": 2}


def fold_plan(model, single_thread=False):
    """Return (workers, threads per worker) for training the folds of `model`.
    Single-threaded models (logreg, svm, etc.) use one fold per core."""
    total = n_jobs()
    if single_thread:
        return total, 1
    family = model.split("_")[0]
    if device() == "cuda" and family in ("xgb", "catboost"):
        return 1, total  # One GPU: keep folds sequential.
    workers = min(FOLD_WORKERS.get(family, 1), total)
    return workers, max(1, total // workers)


def _with_threads(threads, fn, args):
    os.environ["POKEMON_N_JOBS"] = str(threads)  # Read by model factories inside the worker
    return fn(*args)


def fold_map(fn, arg_list, workers, threads):
    """Evaluate [fn(*a) for a in arg_list] with `workers` processes and `threads` threads each (joblib/loky)."""
    if workers <= 1:
        return [fn(*a) for a in arg_list]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=min(workers, len(arg_list)))(delayed(_with_threads)(threads, fn, a) for a in arg_list)


def describe():
    return f"platform={sys.platform} n_jobs={n_jobs()} device={device()} mp={mp_context().get_start_method()}"
