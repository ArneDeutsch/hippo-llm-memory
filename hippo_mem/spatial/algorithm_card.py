"""Algorithmic overview for spatial/procedural memory (SMPD).

Summary
-------
Concise reference for the PlaceGraph map, planning utilities, macro library,
and tool endpoints used by the spatial/procedural memory module.  This module
has no executable code; it serves as a documentation anchor for developers.
Complexity
----------
Not applicable.

Examples
--------
>>> from hippo_mem.spatial import algorithm_card  # doctest: +ELLIPSIS
<module 'hippo_mem.spatial.algorithm_card'...>

See Also
--------
hippo_mem.spatial.map
hippo_mem.spatial.macros

PlaceGraph
==========
Nodes represent unique textual contexts encoded deterministically; repeated
observations reuse existing nodes instead of creating new ones.  Edges store a
``cost`` (float, arbitrary units) and ``success_prob`` (0–1).  Map maintenance
may decay coordinates or prune nodes when their ``last_seen`` exceeds a TTL.

Planners
========
*A\** employs a Euclidean heuristic over pseudo coordinates; **Dijkstra** is the
cost-baseline.  With non‑negative edge costs both guarantee optimal routes and
are used to check one another in tests.

Macro Library
=============
``MacroLib`` records trajectories as macros, tracks success counts, and ranks
suggestions by a smoothed success rate favouring shorter plans.

Tool Endpoints
==============
``SPATIAL.WRITE``
    ``{"context": str}`` → ``{"node_id": int}`` — add or update a place.
``SPATIAL.READ``
    ``{"context": str}`` → ``{"neighbors": list[str]}`` — inspect local graph.
``SPATIAL.PLAN``
    ``{"start": str, "goal": str}`` → ``{"path": list[str], "cost": float}``.
"""

__all__: list[str] = []
