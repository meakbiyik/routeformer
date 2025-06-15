"""Base class for configuration files."""

import copy
from argparse import Namespace


class BaseConfig(Namespace):
    """Base class for configuration files."""

    def __getitem__(self, item):
        """Get item."""
        return getattr(self, item)

    def get(self, item, default):
        """Get item with default value."""
        return getattr(self, item, default)

    def __copy__(self):
        """Copy."""
        return copy.deepcopy(self)

    def copy(self):
        """Copy."""
        return copy.deepcopy(self)

    def override(self, **kwargs):
        """Override config."""
        copy_self = self.copy()
        for k, v in kwargs.items():
            setattr(copy_self, k, v)
        if hasattr(copy_self, "__post_init__"):
            copy_self.__post_init__()
        return copy_self
