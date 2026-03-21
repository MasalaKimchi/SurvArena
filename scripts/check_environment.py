from __future__ import annotations

import platform
import sys


def _print_header() -> None:
    print("SurvArena environment check")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")


def _check_imports() -> None:
    import importlib

    required = [
        "numpy",
        "pandas",
        "yaml",
        "torch",
        "torchsurv",
        "optuna",
        "lifelines",
        "sksurv",
    ]
    missing: list[str] = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        raise RuntimeError(f"Missing required modules: {missing}")
    print("imports=ok")


def _check_torchsurv_metrics() -> None:
    import torch
    from torchsurv.metrics.auc import Auc
    from torchsurv.metrics.brier_score import BrierScore
    from torchsurv.metrics.cindex import ConcordanceIndex
    from torchsurv.stats.ipcw import get_ipcw

    event_train = torch.tensor([True, False, True, False, True, True])
    time_train = torch.tensor([2.0, 3.0, 4.0, 7.0, 8.0, 10.0], dtype=torch.float32)
    event_eval = torch.tensor([True, False, True, False])
    time_eval = torch.tensor([1.5, 4.5, 6.0, 9.0], dtype=torch.float32)
    risk_eval = torch.tensor([0.9, 0.2, 0.6, 0.1], dtype=torch.float32)
    new_time = torch.tensor([2.0, 5.0, 8.0], dtype=torch.float32)

    ipcw_eval = get_ipcw(event_train, time_train, time_eval)
    ipcw_new_time = get_ipcw(event_train, time_train, new_time)

    cindex = ConcordanceIndex()
    uno = cindex(risk_eval, event_eval, time_eval, weight=ipcw_eval)
    harrell = cindex(risk_eval, event_eval, time_eval)

    survival_eval = torch.sigmoid(torch.randn((len(time_eval), len(new_time))))
    brier = BrierScore()
    _ = brier(
        survival_eval,
        event_eval,
        time_eval,
        new_time=new_time,
        weight=ipcw_eval,
        weight_new_time=ipcw_new_time,
    )
    ibs = brier.integral()

    auc = Auc()
    auc_values = auc(
        risk_eval,
        event_eval,
        time_eval,
        new_time=new_time,
        weight=ipcw_eval,
        weight_new_time=ipcw_new_time,
    )
    print(f"torchsurv.uno_c={float(uno):.6f}")
    print(f"torchsurv.harrell_c={float(harrell):.6f}")
    print(f"torchsurv.ibs={float(ibs):.6f}")
    print(f"torchsurv.auc_values={[float(x) for x in auc_values]}")


def main() -> None:
    _print_header()
    _check_imports()
    _check_torchsurv_metrics()
    print("environment_check=passed")


if __name__ == "__main__":
    main()
