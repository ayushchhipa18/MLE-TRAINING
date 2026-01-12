import pytest


@pytest.fixture(autouse=True)
def disable_model_loading(monkeypatch):
    monkeypatch.setenv("PREPROCESSOR_PATH", "dummy.pkl")
    monkeypatch.setenv("MODEL_PATH", "dummy.pkl")
