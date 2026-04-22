from __future__ import annotations

from contextlib import contextmanager
import warnings


@contextmanager
def quiet_training_output(enabled: bool = True):
    if not enabled:
        yield
        return

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
