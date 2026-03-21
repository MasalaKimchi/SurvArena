from __future__ import annotations

from contextlib import contextmanager
import warnings


@contextmanager
def quiet_training_output(enabled: bool = True):
    if not enabled:
        yield
        return

    optuna_logging = None
    previous_verbosity = None
    try:
        import optuna

        optuna_logging = optuna.logging
        previous_verbosity = optuna_logging.get_verbosity()
        optuna_logging.set_verbosity(optuna_logging.WARNING)
    except Exception:
        optuna_logging = None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="all coefficients are zero, consider decreasing alpha.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*X does not have valid feature names.*",
                category=UserWarning,
            )
            yield
    finally:
        if optuna_logging is not None and previous_verbosity is not None:
            optuna_logging.set_verbosity(previous_verbosity)
