"""Phase 0 check: versions, GPU status, TabPFN smoke test on BreastCancer (test set, CPU); logs env to results/env/."""
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from p012.config import load_config, path  # noqa: E402
from p012.data import load_dataset  # noqa: E402
from p012.metrics import auc  # noqa: E402


def main():
    import optuna, pandas, scipy, sklearn, tabpfn, torch, xgboost  # noqa: E401
    from importlib.metadata import version

    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                "tabpfn": version("tabpfn"), "optuna": optuna.__version__, "torch": torch.__version__}
    for k, v in versions.items():
        print(f"{k:13s} {v}")
    pinned = {"xgboost": "2.0.3", "tabpfn": load_config()["tabpfn"]["version"], "scikit-learn": "1.3.2",
              "optuna": "3.5.0"}
    bad = {k: (versions[k], v) for k, v in pinned.items() if versions[k] != v}
    # numpy/scipy: the Mac env keeps llmboost-run's builds (see docs/DEVIATIONS.md); only numpy < 2 matters
    if int(np.__version__.split(".")[0]) >= 2:
        bad["numpy"] = (np.__version__, "<2")

    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    gpu = f"cuda ({torch.cuda.get_device_name(0)})" if cuda else ("none (MPS present, unused)" if mps else "none")
    print(f"GPU           {gpu}  -> TabPFN on {'cuda' if cuda else 'cpu'}")

    # Smoke test: TabPFN fit on the BC pool (455 rows), AUC on the shipped 114-row test set, CPU.
    from tabpfn import TabPFNClassifier
    ds = load_dataset(load_config()["datasets"]["primary"])
    clf = TabPFNClassifier(device="cpu", N_ensemble_configurations=load_config()["tabpfn"]["n_ensemble"], seed=0)
    clf.fit(ds.X_pool.to_numpy(), ds.y_pool, overwrite_warning=True)
    p = clf.predict_proba(ds.X_test.to_numpy())
    smoke = auc(ds.y_test, p)
    print(f"TabPFN smoke  BreastCancer test AUC (CPU) = {smoke:.4f}")

    env = path("results") / "env"
    env.mkdir(parents=True, exist_ok=True)
    (env / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                                       capture_output=True, text=True).stdout)
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
        if sys.platform == "darwin" else platform.processor()
    nvsmi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout if cuda else "no NVIDIA GPU\n"
    (env / "nvidia_smi.txt").write_text(nvsmi)
    (env / "system.txt").write_text(
        "".join(f"{k}: {v}\n" for k, v in {**versions, "os": platform.platform(), "cpu": cpu,
                                           "logical_cores": os.cpu_count(), "gpu": gpu,
                                           "tabpfn_smoke_auc_bc_test_cpu": f"{smoke:.4f}"}.items()))
    print(f"env logged to {env}")

    if bad:
        print("version mismatch:", bad)
    ok = not bad and smoke > 0.9
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
