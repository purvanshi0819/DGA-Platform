import sys, os
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config
from app.ml import dataset


@pytest.fixture(scope="session")
def raw():
    return dataset.load_raw(config.DATASET_PATH)
