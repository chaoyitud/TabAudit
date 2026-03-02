from __future__ import annotations

from tab_audit.modeling.backend import select_backend


def test_backend_fallback_when_gpu_libs_missing(monkeypatch):
    monkeypatch.setattr("tab_audit.modeling.backend.detect_cuda_available", lambda: True)

    def fake_available(_: str) -> bool:
        return False

    monkeypatch.setattr("tab_audit.modeling.backend._module_available", fake_available)

    backend, device, warnings = select_backend("auto", ["xgboost", "lightgbm", "catboost", "sklearn"])
    assert backend == "sklearn"
    assert device == "cpu"
    assert any("falling back" in w.lower() for w in warnings)
