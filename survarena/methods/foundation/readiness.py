from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path

from survarena.methods.foundation.catalog import FoundationModelSpec, available_foundation_model_specs


_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
_FOUNDATION_SPEC_BY_ID = {spec.method_id: spec for spec in available_foundation_model_specs()}


@dataclass(frozen=True, slots=True)
class FoundationRuntimeStatus:
    method_id: str
    dependency_module: str | None
    install_extra: str | None
    dependency_installed: bool
    runtime_ready: bool
    requires_hf_auth: bool
    auth_configured: bool | None
    install_command: str | None
    blocked_reason: str | None = None
    warning_reason: str | None = None


def _install_command(spec: FoundationModelSpec) -> str | None:
    if spec.install_extra is None:
        return None
    return f'python -m pip install -e ".[{spec.install_extra}]"'


def _has_dependency(module_name: str | None) -> bool:
    if module_name is None:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _huggingface_auth_configured() -> bool:
    for env_var in _HF_TOKEN_ENV_VARS:
        value = os.environ.get(env_var)
        if value and value.strip():
            return True
    try:
        hub = importlib.import_module("huggingface_hub")
    except Exception:
        return False

    get_token = getattr(hub, "get_token", None)
    if callable(get_token):
        try:
            token = get_token()
            if token:
                return True
        except Exception:
            pass

    hf_folder = getattr(hub, "HfFolder", None)
    if hf_folder is not None and hasattr(hf_folder, "get_token"):
        try:
            token = hf_folder.get_token()
            if token:
                return True
        except Exception:
            pass
    return False


def foundation_runtime_status(
    spec: FoundationModelSpec,
    *,
    checkpoint_path: str | None = None,
) -> FoundationRuntimeStatus:
    dependency_installed = _has_dependency(spec.dependency_module)
    install_command = _install_command(spec)
    blocked_reason: str | None = None
    warning_reason: str | None = None
    auth_configured: bool | None = None

    if not dependency_installed:
        blocked_reason = (
            f"Dependency '{spec.dependency_module}' is not installed."
            + (f" Install it with `{install_command}`." if install_command is not None else "")
        )
        return FoundationRuntimeStatus(
            method_id=spec.method_id,
            dependency_module=spec.dependency_module,
            install_extra=spec.install_extra,
            dependency_installed=False,
            runtime_ready=False,
            requires_hf_auth=spec.requires_hf_auth,
            auth_configured=auth_configured,
            install_command=install_command,
            blocked_reason=blocked_reason,
        )

    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            blocked_reason = f"Configured checkpoint_path '{checkpoint}' does not exist."
            return FoundationRuntimeStatus(
                method_id=spec.method_id,
                dependency_module=spec.dependency_module,
                install_extra=spec.install_extra,
                dependency_installed=True,
                runtime_ready=False,
                requires_hf_auth=spec.requires_hf_auth,
                auth_configured=auth_configured,
                install_command=install_command,
                blocked_reason=blocked_reason,
            )

    if spec.requires_hf_auth and checkpoint_path is None:
        auth_configured = _huggingface_auth_configured()
        if not auth_configured:
            warning_reason = (
                "Hugging Face authentication was not detected. If gated TabPFN weights are not already cached locally, "
                "the first run may fail. Run `hf auth login` or set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`."
            )

    return FoundationRuntimeStatus(
        method_id=spec.method_id,
        dependency_module=spec.dependency_module,
        install_extra=spec.install_extra,
        dependency_installed=dependency_installed,
        runtime_ready=blocked_reason is None,
        requires_hf_auth=spec.requires_hf_auth,
        auth_configured=auth_configured,
        install_command=install_command,
        blocked_reason=blocked_reason,
        warning_reason=warning_reason,
    )


def foundation_runtime_catalog() -> tuple[FoundationRuntimeStatus, ...]:
    return tuple(foundation_runtime_status(spec) for spec in available_foundation_model_specs())


def foundation_runtime_status_for_method(
    method_id: str,
    *,
    checkpoint_path: str | None = None,
) -> FoundationRuntimeStatus:
    if method_id not in _FOUNDATION_SPEC_BY_ID:
        raise ValueError(f"Unknown foundation method_id '{method_id}'.")
    return foundation_runtime_status(_FOUNDATION_SPEC_BY_ID[method_id], checkpoint_path=checkpoint_path)


def ensure_foundation_runtime_ready(
    method_id: str,
    *,
    checkpoint_path: str | None = None,
) -> FoundationRuntimeStatus:
    status = foundation_runtime_status_for_method(method_id, checkpoint_path=checkpoint_path)
    if not status.runtime_ready:
        raise RuntimeError(status.blocked_reason or f"Foundation model '{method_id}' is not ready.")
    return status


def rewrite_foundation_runtime_error(
    method_id: str,
    exc: Exception,
    *,
    checkpoint_path: str | None = None,
) -> RuntimeError:
    status = foundation_runtime_status_for_method(method_id, checkpoint_path=checkpoint_path)
    message = str(exc)
    lower_message = message.lower()
    if status.blocked_reason is not None and message == status.blocked_reason:
        return RuntimeError(message)
    if method_id == "tabpfn_survival" and checkpoint_path is None:
        if any(token in lower_message for token in ("huggingface", "gated", "401", "403", "tabpfn_2_5")):
            return RuntimeError(
                "TabPFN access is not ready for the default gated checkpoint. "
                "Accept the model terms at https://huggingface.co/Prior-Labs/tabpfn_2_5 and authenticate with "
                "`hf auth login` or `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`. "
                "You can also point `checkpoint_path` at a local checkpoint."
            )
    if "No module named" in message and status.install_command is not None:
        return RuntimeError(
            f"{method_id} is not installed correctly. "
            f"Install the required extra with `{status.install_command}`."
        )
    prefix = status.blocked_reason or status.warning_reason
    if prefix:
        return RuntimeError(f"{prefix} Original error: {message}")
    return RuntimeError(message)
