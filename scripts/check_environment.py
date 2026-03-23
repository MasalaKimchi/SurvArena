from __future__ import annotations

import argparse
import platform
import sys


CORE_REQUIRED = [
    "numpy",
    "pandas",
    "yaml",
    "torch",
    "torchsurv",
    "optuna",
    "lifelines",
    "sksurv",
]

FOUNDATION_REQUIRED = [
    "tabpfn",
    "autogluon.tabular",
]


def _print_header() -> None:
    print("SurvArena environment check")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"executable={sys.executable}")


def _check_virtualenv() -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        print(f"venv=ok ({sys.prefix})")
        return
    print("venv=warning (global interpreter detected; prefer a repo-local .venv to avoid dependency conflicts)")


def _check_imports(required: list[str], *, label: str) -> None:
    import importlib

    missing: list[str] = []
    for pkg in required:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, "__version__", "unknown")
            print(f"import[{pkg}]={version}")
        except Exception:
            missing.append(pkg)
    if missing:
        raise RuntimeError(f"Missing required modules for {label}: {missing}")
    print(f"{label}=ok")


def _check_torchsurv_metrics() -> None:
    import torch
    from torchsurv.loss.cox import neg_partial_log_likelihood
    from torchsurv.loss.momentum import Momentum
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
    _ = neg_partial_log_likelihood(torch.tensor([0.1, 0.2]), torch.tensor([True, False]), torch.tensor([1.0, 2.0]))
    _ = Momentum
    print(f"torchsurv.uno_c={float(uno):.6f}")
    print(f"torchsurv.harrell_c={float(harrell):.6f}")
    print(f"torchsurv.ibs={float(ibs):.6f}")
    print(f"torchsurv.auc_values={[float(x) for x in auc_values]}")
    print("torchsurv.losses=ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local SurvArena environment.")
    parser.add_argument(
        "--include-foundation",
        action="store_true",
        help="Also validate optional foundation-model dependencies.",
    )
    args = parser.parse_args()

    _print_header()
    _check_virtualenv()
    _check_imports(CORE_REQUIRED, label="core_imports")
    if args.include_foundation:
        _check_imports(FOUNDATION_REQUIRED, label="foundation_imports")
    else:
        print("foundation_imports=skipped")
    _check_torchsurv_metrics()
    print("environment_check=passed")


if __name__ == "__main__":
    main()
