from __future__ import annotations

from survarena.methods.foundation.catalog import available_foundation_model_specs
from survarena.methods.foundation.readiness import foundation_runtime_status


def test_foundation_runtime_status_reports_install_command_for_missing_dependency(monkeypatch) -> None:
    spec = next(item for item in available_foundation_model_specs() if item.method_id == "mitra_survival")

    monkeypatch.setattr("survarena.methods.foundation.readiness._has_dependency", lambda module_name: False)

    status = foundation_runtime_status(spec)

    assert status.runtime_ready is False
    assert status.dependency_installed is False
    assert status.install_extra == "foundation-mitra"
    assert status.install_command == 'python -m pip install -e ".[foundation-mitra]"'
    assert status.blocked_reason is not None


def test_foundation_runtime_status_warns_when_tabpfn_auth_is_missing(monkeypatch) -> None:
    spec = next(item for item in available_foundation_model_specs() if item.method_id == "tabpfn_survival")

    monkeypatch.setattr("survarena.methods.foundation.readiness._has_dependency", lambda module_name: True)
    monkeypatch.setattr("survarena.methods.foundation.readiness._huggingface_auth_configured", lambda: False)

    status = foundation_runtime_status(spec)

    assert status.runtime_ready is True
    assert status.auth_configured is False
    assert status.warning_reason is not None
    assert "hf auth login" in status.warning_reason
