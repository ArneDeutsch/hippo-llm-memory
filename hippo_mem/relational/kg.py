"""Knowledge graph helpers."""

from typing import Dict, List


class KnowledgeGraph:
    """Minimal knowledge graph stub."""

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self.edges: Dict[str, List[str]] = {}

    def add_edge(self, src: str, dst: str) -> None:
        """Add an edge to the graph."""
        self.edges.setdefault(src, []).append(dst)
