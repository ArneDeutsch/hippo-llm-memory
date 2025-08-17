"""Spatial and procedural memory package.

Summary
-------
Provides the spatial/procedural memory (SMPD) components: deterministic
``PlaceGraph`` maps, a ``MacroLib`` of reusable trajectories, and
``SpatialAdapter`` for cross-attention with plan embeddings.

Parameters
----------
None

Returns
-------
None

Raises
------
None

Side Effects
------------
Importing submodules may allocate small caches.

Complexity
----------
Not applicable.

Examples
--------
>>> from hippo_mem.spatial import PlaceGraph, MacroLib

See Also
--------
hippo_mem.spatial.algorithm_card
hippo_mem.spatial.map
hippo_mem.spatial.macros
"""

from .adapter import AdapterConfig, SpatialAdapter
from .macros import Macro, MacroLib
from .map import PlaceGraph

__all__ = [
    "AdapterConfig",
    "SpatialAdapter",
    "Macro",
    "MacroLib",
    "PlaceGraph",
]
