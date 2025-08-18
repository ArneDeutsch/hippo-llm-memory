"""SGC-RSS relational memory package.

Summary
-------
Central package implementing Schema-Guided Consolidation with a
Relational Semantic Store (SGC-RSS). Text is converted into
``(head, relation, tail)`` tuples, routed via a ``SchemaIndex`` and
persisted as a ``SemanticGraph`` backed by SQLite/NetworkX. A
dual-path adapter deterministically fuses subgraph features with
episodic traces, and schema-fit tuples are fast-tracked to the graph
for quick consolidation.
Complexity
----------
All components are designed for small graphs (≤10^4 edges) and run in
milliseconds on CPU.

Examples
--------
>>> from hippo_mem.relational import tuples
>>> tuples.extract_tuples("Alice likes Bob.")
[('Alice', 'likes', 'Bob', 'Alice likes Bob.', None, 0.666..., 0)]

See Also
--------
hippo_mem.relational.tuples : Tuple extraction utilities.
hippo_mem.relational.schema : Schema prototypes and routing.
hippo_mem.relational.kg : SemanticGraph persistence and retrieval.
hippo_mem.relational.adapter : Dual-path fusion adapter.
"""

__all__ = ["adapter", "kg", "schema", "tuples"]
