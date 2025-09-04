import torch
from torch import nn

from hippo_mem.common import TraceSpec
from hippo_mem.spatial.place_graph import PlaceGraph
from hippo_mem.spatial.retrieval import spatial_retrieve_and_pack


def test_subgraph_packing_and_mask() -> None:
    g = PlaceGraph()
    for ctx in ["a", "b", "c"]:
        g.observe(ctx)

    spec = TraceSpec(
        source="spatial",
        params={"radius": 1, "max_nodes": 4, "max_edges": 3},
    )
    proj = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(4))

    mem = spatial_retrieve_and_pack("b", spec, g, proj)
    assert mem.tokens.shape == (1, 7, 4)
    assert mem.mask.shape == (1, 7)
    assert mem.mask[0, :5].all()
    assert not mem.mask[0, 5:].any()
    assert mem.meta["hint"] == "go to a"

    nodes, edges = g.local("b", radius=1)
    feats = []
    for n in nodes:
        ctx = g._id_to_context[n]
        p = g.encoder.encode(ctx)
        feats.append(torch.tensor([p.coord[0], p.coord[1], float(p.last_seen), 1.0]))
    for u, v, _ in edges:
        e = g.graph[u][v]
        feats.append(torch.tensor([e.cost, e.success, float(e.last_seen), 0.0]))
    expected = torch.stack(feats).unsqueeze(0)
    assert torch.allclose(mem.tokens[:, :5], expected, atol=0, rtol=0)
