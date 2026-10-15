"""Every test uses its own store; never touch the learner's real experiments."""
import pytest

@pytest.fixture(autouse=True)
def isolated_experiment_store(tmp_path,monkeypatch):
    monkeypatch.setenv('CNN_LAB_EXPERIMENT_DIR',str(tmp_path/'experiments'))
