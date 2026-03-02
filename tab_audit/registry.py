from __future__ import annotations

from tab_audit.connectors.kaggle_connector import fetch_kaggle_dataset
from tab_audit.connectors.local_connector import fetch_local_dataset
from tab_audit.connectors.openml_connector import fetch_openml_dataset

CONNECTOR_REGISTRY = {
    "local": fetch_local_dataset,
    "openml": fetch_openml_dataset,
    "kaggle": fetch_kaggle_dataset,
}
