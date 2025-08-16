import copy

import torch

from hippo_mem.spatial.adapter import AdapterConfig, SpatialAdapter
from hippo_mem.spatial.macros import MacroLib
from hippo_mem.spatial.map import PlaceGraph


def test_graph_growth_deterministic() -> None:
    seq = ["a", "b", "c", "a"]
    g1 = PlaceGraph()
    for s in seq:
        g1.observe(s)
    g2 = PlaceGraph()
    for s in seq:
        g2.observe(s)

    assert g1.graph == g2.graph
    assert g1._context_to_id == g2._context_to_id
    enc = g1.encoder
    assert enc.encode("a").last_seen == 4
    assert enc.encode("b").last_seen == 2
    a = g1._context_to_id["a"]
    b = g1._context_to_id["b"]
    assert g1.graph[a][b].last_seen == 2


def test_path_integration_planning() -> None:
    seq = ["start", "mid", "goal"]
    g_no = PlaceGraph(path_integration=False)
    for s in seq:
        g_no.observe(s)
    g_pi = PlaceGraph(path_integration=True)
    for s in seq:
        g_pi.observe(s)

    path_a = g_pi.plan("start", "goal", method="astar")
    path_d = g_pi.plan("start", "goal", method="dijkstra")
    assert path_a == path_d == seq
    assert g_pi.encoder.encode("start").coord == (0.0, 0.0)
    assert g_pi.encoder.encode("mid").coord != g_no.encoder.encode("mid").coord


def test_macro_replay_improves_success() -> None:
    lib = MacroLib()
    lib.store("bad", ["s", "g"])
    lib.store("good", ["s", "g"])

    baseline = lib.suggest("s", "g", k=1)[0].name
    assert baseline == "bad"

    lib.update_stats("good", True)
    lib.update_stats("good", True)
    lib.update_stats("bad", False)
    lib.update_stats("bad", False)

    improved = lib.suggest("s", "g", k=1)[0].name
    assert improved == "good"


def test_spatial_adapter_integration() -> None:
    """SpatialAdapter fuses hidden states with plan embeddings."""

    g = PlaceGraph()
    for ctx in ["s", "m", "g"]:
        g.observe(ctx)
    path = g.plan("s", "g")

    lib = MacroLib()
    lib.store("route", path)
    macro = lib.suggest("s", "g", k=1)[0]

    emb = [g.encoder.encode(c).coord for c in macro.trajectory]
    plan = (
        torch.tensor([[x, y, 0.0, 0.0] for x, y in emb], dtype=torch.float32)
        .unsqueeze(0)
        .requires_grad_()
    )
    hidden = torch.randn(1, 1, 4, requires_grad=True)

    adapter = SpatialAdapter(AdapterConfig(hidden_size=4, num_heads=2))
    out = adapter(hidden, plan)
    assert out.shape == hidden.shape
    out.sum().backward()
    assert hidden.grad is not None and plan.grad is not None


def test_placegraph_maintenance_and_rollback() -> None:
    seq = ["start", "mid", "goal"]
    g = PlaceGraph(path_integration=True)
    for s in seq:
        g.observe(s)

    coords_before = {s: g.encoder.encode(s).coord for s in seq}
    graph_before = copy.deepcopy(g.graph)

    g.decay(0.5)
    assert g.encoder.encode("mid").coord != coords_before["mid"]

    start_id = g._context_to_id["start"]
    g.prune(max_age=1)
    assert "start" not in g._context_to_id

    g.rollback(2)
    restored = {s: g.encoder.encode(s).coord for s in seq}
    assert restored == coords_before
    assert g.graph == graph_before
    assert set(g._context_to_id.keys()) == set(seq)
    assert g._context_to_id["start"] == start_id
