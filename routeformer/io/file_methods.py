"""PLData file reader.

Sourced from https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/file_methods.py. # noqa: E501

For the LICENSE, see https://github.com/pupil-labs/pupil/blob/master/COPYING.

CHANGES:
- Removed unused imports.
- Removed unused functions.
- Removed module-level variables and function calls.
- Removed prints and benchmarking.
- Removed rich.progress dependency
- Removed writer class.
- Formatted with black.
- Improved documentation and type hints.

(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import collections.abc
import logging
import os
import pickle
import types
from pathlib import Path

import msgpack
import numpy as np

assert msgpack.version[0] == 1, "msgpack out of date, please upgrade to version (1, 0, 0)"


logger = logging.getLogger(__name__)
UnpicklingError = pickle.UnpicklingError

PLData = collections.namedtuple("PLData", ["data", "timestamps", "topics"])


def load_object(file_path):
    """Load a Python object with msgpack from a file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    object
        The loaded object.
    """
    import gc

    file_path = Path(file_path).expanduser()
    with file_path.open("rb") as fh:
        gc.disable()  # speeds deserialization up.
        data = msgpack.unpack(fh, strict_map_key=False)
        gc.enable()
    return data


def load_pldata_file(directory: str, topic: str) -> PLData:
    """Load a PLData file.

    Parameters
    ----------
    directory : str
        The directory containing the PLData file.
    topic : str
        The topic of the PLData file.

    Returns
    -------
    PLData
        The loaded PLData.
    """
    ts_file = os.path.join(directory, topic + "_timestamps.npy")
    msgpack_file = os.path.join(directory, topic + ".pldata")
    data = collections.deque()
    topics = collections.deque()
    data_ts = np.load(ts_file)
    with open(msgpack_file, "rb") as fh:
        unpacker = msgpack.Unpacker(fh, use_list=False, strict_map_key=False)
        for topic, payload in unpacker:
            datum = Serialized_Dict(msgpack_bytes=payload)
            data.append(datum)
            topics.append(topic)

    return PLData(data, data_ts, topics)


class _Empty:
    def purge_cache(self):
        pass


class Serialized_Dict:
    """A dict-like object that deserializes msgpack data on demand."""

    __slots__ = ["_ser_data", "_data"]
    cache_len = 100
    _cache_ref = [_Empty()] * cache_len
    MSGPACK_EXT_CODE = 13

    @classmethod
    def packing_hook(self, obj):
        """Pack msgpack data."""
        if isinstance(obj, self):
            return msgpack.ExtType(self.MSGPACK_EXT_CODE, obj.serialized)
        raise TypeError(f"can't serialize {type(obj)}({repr(obj)})")

    def __init__(self, msgpack_bytes=None):
        """Initialize a Serialized_Dict."""
        if type(msgpack_bytes) is bytes:
            self._ser_data = msgpack_bytes
        self._data = None

    def _deser(self):
        if not self._data:
            self._data = msgpack.unpackb(
                self._ser_data,
                use_list=False,
                object_hook=self.unpacking_object_hook,
                strict_map_key=False,
            )
            self._cache_ref.pop(0).purge_cache()
            self._cache_ref.append(self)

    @classmethod
    def unpacking_object_hook(self, obj):
        """Unpack msgpack data."""
        if type(obj) is dict:
            return types.MappingProxyType(obj)

    def purge_cache(self):
        """Purge the cache."""
        self._data = None

    def __getitem__(self, key):
        """Get an item from the dict-like object."""
        self._deser()
        return self._data[key]
