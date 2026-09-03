from __future__ import annotations

import numpy as np
import pytest
import torch

from survarena.evaluation.concordance_time_dependent import (
    antolini_c_index,
    harrell_c_index,
    time_dependent_uno_c_index,
)
from survarena.methods.deep.rsbs_loss import (
    censored_log_intensity_score,
    kmd_u_statistic,
    reference_scaled_bregman_score,
    rsbs_loss,
)


def test_beta_one_matches_log_score_up_to_reference_constant() -> None:
    q = torch.tensor([[0.2, 0.6], [0.5, 0.9]], dtype=torch.float64)
    e = torch.tensor([[1.0, 0.4], [1.0, 0.8]], dtype=torch.float64)
    d = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    a = torch.tensor([0.3, 0.7], dtype=torch.float64)
    log_score = censored_log_intensity_score(q, e, d, reduction="none")
    beta_one = reference_scaled_bregman_score(q, e, d, a, beta=1, reduction="none")
    constant = (d * torch.log(a.expand_as(q))).sum(dim=-1)
    torch.testing.assert_close(beta_one, log_score + constant)


def test_reference_scaled_score_is_time_unit_invariant() -> None:
    q = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    e = torch.tensor([[3.0, 2.0]], dtype=torch.float64)
    d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    a = torch.tensor([0.3, 0.5], dtype=torch.float64)
    scale = 365.25
    original = reference_scaled_bregman_score(q, e, d, a, beta=1.5)
    transformed = reference_scaled_bregman_score(q / scale, e * scale, d, a / scale, beta=1.5)
    torch.testing.assert_close(original, transformed, rtol=1e-12, atol=1e-12)


def test_expected_quadratic_score_minimized_at_truth() -> None:
    truth = 0.7
    candidates = torch.linspace(0.05, 1.4, 200, dtype=torch.float64)
    losses = torch.stack([
        reference_scaled_bregman_score(
            candidate.reshape(1, 1),
            torch.ones((1, 1), dtype=torch.float64),
            torch.tensor([[truth]], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            beta=2,
        )
        for candidate in candidates
    ])
    assert float(candidates[int(torch.argmin(losses))]) == pytest.approx(truth, abs=0.01)


def test_rsbs_detaches_reference_and_u_statistic_removes_diagonal() -> None:
    q = torch.tensor([[0.4]], requires_grad=True)
    a = torch.tensor([0.3], requires_grad=True)
    rsbs_loss(q, torch.tensor([[1.0]]), torch.tensor([[1.0]]), a, eta=0.2).backward()
    assert q.grad is not None and a.grad is None
    assert float(kmd_u_statistic(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))) == pytest.approx(-0.5)


def test_concordance_reference_cases() -> None:
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0], dtype=bool)
    risk = np.array([3.0, 2.0, 1.0])
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    survival = np.array([[1.0, 0.5, 0.3, 0.2], [1.0, 0.8, 0.6, 0.4], [1.0, 0.9, 0.8, 0.7]])
    assert harrell_c_index(time, event, risk) == 1.0
    assert antolini_c_index(time, event, survival, grid) == 1.0
    assert time_dependent_uno_c_index(
        time, event, survival, grid, censor_survival=np.ones(3)
    ) == 1.0
