"""routeformer package index."""
from routeformer.io import DreyeveDataset, GEMDataset
from routeformer.models import Routeformer
from routeformer.utils import set_logger_config

__all__ = [
    "GEMDataset",
    "DreyeveDataset",
    "set_logger_config",
    "Routeformer",
]

# set the logger config per environment variables
set_logger_config()
