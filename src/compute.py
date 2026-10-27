"""Configurazione hardware portabile (macOS / Linux / Windows).

Tutto è sovrascrivibile da variabili d'ambiente o dalle opzioni --n-jobs / --device degli script:
    POKEMON_N_JOBS   numero di processi/thread (default: tutti i core)
    POKEMON_DEVICE   auto | cpu | cuda (default: auto -> cuda se c'è una GPU NVIDIA)
    POKEMON_MP       fork | spawn (default: fork dove esiste, spawn su Windows)

- Multiprocessing: `fork` su Linux/macOS (i worker ereditano i dati del padre), `spawn` su
  Windows (dove `fork` non esiste); i dati passano dall'initializer, quindi funziona con entrambi.
- GPU: XGBoost (`device="cuda"`) e CatBoost (`task_type="GPU"`) solo con CUDA (Linux/Windows
  con NVIDIA). Su Mac Apple Silicon non c'è backend GPU per questi modelli: si usa la CPU.
- LightGBM resta su CPU (la build GPU non è quella standard di pip/conda).
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
    forced = os.environ.get("POKEMON_MP")  # fork | spawn, per testare il percorso Windows altrove
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
    """Applica le opzioni da riga di comando (prima di creare modelli o pool)."""
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
    """map parallelo portabile; con un solo job gira nel processo corrente."""
    if n_jobs() == 1:
        if initializer:
            initializer(*initargs)
        return [func(x) for x in items]
    with mp_context().Pool(n_jobs(), initializer=initializer, initargs=initargs) as pool:
        return pool.map(func, items, chunksize=chunksize)


# Fold allenati in parallelo per famiglia di modelli; i thread di ciascuno sono n_jobs // worker.
# Misurato sul M5 (4 core P + 6 core E) con 10k righe: i boosting scalano male oltre ~3 thread
# (lgbm/xgb 1.5x più veloci con 3 worker x 3 thread che con 1 x 10, catboost 1.1x con 2 x 5);
# ExtraTrees/RF di sklearn scalano bene sui thread e restano 1 worker x tutti i thread.
FOLD_WORKERS = {"lgbm": 3, "xgb": 3, "catboost": 2}


def fold_plan(model, single_thread=False):
    """(worker, thread per worker) per allenare i fold di `model`. single_thread: modelli a un
    thread (logreg, svm, ...), un fold per core."""
    total = n_jobs()
    if single_thread:
        return total, 1
    family = model.split("_")[0]
    if device() == "cuda" and family in ("xgb", "catboost"):
        return 1, total  # una sola GPU: i fold restano sequenziali
    workers = min(FOLD_WORKERS.get(family, 1), total)
    return workers, max(1, total // workers)


def _with_threads(threads, fn, args):
    os.environ["POKEMON_N_JOBS"] = str(threads)  # letto dalle factory dei modelli nel worker
    return fn(*args)


def fold_map(fn, arg_list, workers, threads):
    """[fn(*a) for a in arg_list], con `workers` processi (joblib/loky, portabile) da `threads` thread."""
    if workers <= 1:
        return [fn(*a) for a in arg_list]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=min(workers, len(arg_list)))(delayed(_with_threads)(threads, fn, a) for a in arg_list)


def describe():
    return f"platform={sys.platform} n_jobs={n_jobs()} device={device()} mp={mp_context().get_start_method()}"
