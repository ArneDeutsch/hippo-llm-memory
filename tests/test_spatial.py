import copy
import math
import time

import networkx as nx
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from hippo_mem.adapters.spatial_adapter import SpatialMemoryAdapter
from hippo_mem.common import MemoryTokens
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
    """SpatialAdapter matches manual attention and mask behaviour."""

    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    plan = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        requires_grad=True,
    )

    adapter = SpatialAdapter(AdapterConfig(hidden_size=4, num_heads=2))
    eye = torch.eye(4)
    with torch.no_grad():
        adapter.q_proj.weight.copy_(eye)
        adapter.k_proj.weight.copy_(eye)
        adapter.v_proj.weight.copy_(eye)
        adapter.o_proj.weight.copy_(eye)

    memory = MemoryTokens(tokens=plan, mask=torch.tensor([[True, True]]))
    out = adapter(hidden, memory=memory)

    head_dim = adapter.head_dim
    q = hidden.view(1, 1, adapter.num_heads, head_dim).transpose(1, 2)
    k = plan.view(1, plan.shape[1], adapter.num_heads, head_dim).transpose(1, 2)
    v = k.clone()
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn, v)
    context = context.transpose(1, 2).reshape(1, 1, adapter.hidden_size)
    expected = hidden + context
    assert torch.allclose(out, expected)

    out.sum().backward()
    assert hidden.grad is not None and plan.grad is not None

    empty_mem = MemoryTokens(tokens=plan[:, :0], mask=torch.zeros(1, 0, dtype=torch.bool))
    empty_out = adapter(hidden, memory=empty_mem)
    assert torch.allclose(empty_out, hidden)


def test_expand_kv_noop_when_heads_match() -> None:
    """_expand_kv returns input unchanged when kv and query heads match."""

    cfg = AdapterConfig(hidden_size=8, num_heads=2, num_kv_heads=2)
    adapter = SpatialAdapter(cfg)
    x = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    expanded = adapter._expand_kv(x)
    assert expanded.shape == x.shape
    assert torch.equal(expanded, x)


def test_expand_kv_multi_query_attention() -> None:
    """_expand_kv duplicates K/V heads when ``num_kv_heads=1``."""

    cfg = AdapterConfig(hidden_size=8, num_heads=4, num_kv_heads=1)
    adapter = SpatialAdapter(cfg)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (b=1, kvh=1, t=2, d=2)
    expanded = adapter._expand_kv(x)
    assert expanded.shape == (1, 4, 2, 2)
    for h in range(adapter.num_heads):
        assert torch.equal(expanded[0, h], x[0, 0])


def test_expand_kv_grouped_query() -> None:
    """Grouped-query attention duplicates each KV head."""

    cfg = AdapterConfig(hidden_size=8, num_heads=4, num_kv_heads=2)
    adapter = SpatialAdapter(cfg)
    x = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    expanded = adapter._expand_kv(x)
    assert expanded.shape == (1, 4, 1, 2)
    assert torch.equal(expanded[0, 0], x[0, 0])
    assert torch.equal(expanded[0, 1], x[0, 0])
    assert torch.equal(expanded[0, 2], x[0, 1])
    assert torch.equal(expanded[0, 3], x[0, 1])


def test_spatial_memory_adapter_zero_mask_returns_zero() -> None:
    """SpatialMemoryAdapter returns zeros when mask has no true values."""

    cfg = AdapterConfig(hidden_size=4, num_heads=2)
    adapter = SpatialMemoryAdapter(cfg)
    hidden = torch.randn(1, 1, 4)
    tokens = torch.randn(1, 1, 4)
    memory = MemoryTokens(tokens=tokens, mask=torch.zeros(1, 1, dtype=torch.bool))
    out = adapter(hidden, memory=memory)
    assert torch.allclose(out, torch.zeros_like(hidden))


def test_spatial_memory_adapter_masks_and_grads() -> None:
    """SpatialMemoryAdapter respects masks and backpropagates gradients."""

    cfg = AdapterConfig(hidden_size=4, num_heads=2)
    adapter = SpatialMemoryAdapter(cfg)

    hidden0 = torch.randn(1, 1, 4, requires_grad=True)
    tokens0 = torch.randn(1, 1, 4, requires_grad=True)
    mem0 = MemoryTokens(tokens=tokens0, mask=torch.zeros(1, 1, dtype=torch.bool))
    out0 = adapter(hidden0, memory=mem0)
    assert torch.allclose(out0, torch.zeros_like(hidden0))

    hidden = torch.randn(1, 1, 4, requires_grad=True)
    tokens = torch.randn(1, 1, 4, requires_grad=True)
    mem = MemoryTokens(tokens=tokens, mask=torch.ones(1, 1, dtype=torch.bool))

    out = adapter(hidden, memory=mem)
    with torch.no_grad():
        base_mem = MemoryTokens(
            tokens=tokens.detach().clone(), mask=torch.ones(1, 1, dtype=torch.bool)
        )
        baseline = adapter.inner(hidden.detach().clone(), memory=base_mem)
    assert torch.allclose(hidden + out, baseline)

    out.sum().backward()
    assert hidden.grad is not None
    assert tokens.grad is not None


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


def test_placegraph_maintenance_log_records_events() -> None:
    """Decay and prune operations are logged in the maintenance log."""

    g = PlaceGraph()
    g.observe("a")
    g.observe("b")

    g.decay(0.1)
    g.prune(max_age=0)

    ops = [e["op"] for e in g._maintenance_log]
    assert ops == ["decay", "prune"]
    assert g._maintenance_log[0]["rate"] == 0.1
    assert g._maintenance_log[1]["max_age"] == 0


def test_prune_removes_stale_edges_only() -> None:
    g = PlaceGraph()
    for ctx in ["a", "b", "c"]:
        g.observe(ctx)
    g.connect("a", "c")
    g._step = 10
    for ctx in ["a", "b", "c"]:
        g.encoder.encode(ctx).last_seen = 10
    a, b, c = (g._context_to_id[x] for x in ["a", "b", "c"])
    g.graph[a][b].last_seen = 1
    g.graph[b][a].last_seen = 1
    g.graph[a][c].last_seen = 9
    g.graph[c][a].last_seen = 9
    g.graph[b][c].last_seen = 9
    g.graph[c][b].last_seen = 9

    g.prune(max_age=5)
    assert set(g._context_to_id.keys()) == {"a", "b", "c"}
    assert b not in g.graph[a]
    assert a not in g.graph[b]
    assert c in g.graph[a] and a in g.graph[c]


def test_prune_and_rollback_restore_state() -> None:
    g = PlaceGraph()
    for ctx in ["a", "b", "c"]:
        g.observe(ctx)
    snapshot_graph = copy.deepcopy(g.graph)
    snapshot_ctx = dict(g._context_to_id)

    g._step = 10
    g.encoder.encode("a").last_seen = 1
    g.encoder.encode("b").last_seen = 9
    g.encoder.encode("c").last_seen = 9

    g.prune(max_age=5)
    assert "a" not in g._context_to_id

    g.rollback(1)
    assert g.graph == snapshot_graph
    assert g._context_to_id == snapshot_ctx


@st.composite
def _graph_fixture(draw) -> tuple[PlaceGraph, str, str]:
    n = draw(st.integers(min_value=2, max_value=5))
    g = PlaceGraph()
    ctxs = [str(i) for i in range(n)]
    for c in ctxs:
        g.observe(c)
    for i in range(n - 1):
        a, b = ctxs[i], ctxs[i + 1]
        ca, cb = g.encoder.encode(a).coord, g.encoder.encode(b).coord
        cost = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
        g.connect(a, b, cost=cost)
    for i in range(n - 2):
        for j in range(i + 2, n):
            if draw(st.booleans()):
                a, b = ctxs[i], ctxs[j]
                ca, cb = g.encoder.encode(a).coord, g.encoder.encode(b).coord
                cost = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                g.connect(a, b, cost=cost)
    return g, ctxs[0], ctxs[-1]


@settings(max_examples=25, deadline=None)
@given(_graph_fixture())
def test_planner_astar_matches_dijkstra(data: tuple[PlaceGraph, str, str]) -> None:
    """A* and Dijkstra produce the same optimal path on random DAGs."""

    g, start, goal = data
    path_a = g.plan(start, goal, method="astar")
    path_d = g.plan(start, goal, method="dijkstra")
    assert path_a == path_d

    G = nx.Graph()
    for a, nbrs in g.graph.items():
        for b, edge in nbrs.items():
            G.add_edge(a, b, weight=edge.cost)
    expected = nx.dijkstra_path_length(G, g._context_to_id[start], g._context_to_id[goal])
    cost = 0.0
    for u, v in zip(path_a, path_a[1:]):
        cost += g.graph[g._context_to_id[u]][g._context_to_id[v]].cost
    assert abs(cost - expected) < 1e-6


@settings(max_examples=25, deadline=None)
@given(_graph_fixture())
def test_planner_optimality_property(data: tuple[PlaceGraph, str, str]) -> None:
    """Planner returns a minimal-cost path on random graphs."""

    g, start, goal = data
    path = g.plan(start, goal, method="astar")
    assert path == g.plan(start, goal, method="dijkstra")

    cost = 0.0
    for u, v in zip(path, path[1:]):
        cost += g.graph[g._context_to_id[u]][g._context_to_id[v]].cost

    G = nx.Graph()
    for a, nbrs in g.graph.items():
        for b, edge in nbrs.items():
            G.add_edge(a, b, weight=edge.cost)
    start_id = g._context_to_id[start]
    goal_id = g._context_to_id[goal]
    min_cost = float("inf")
    for p in nx.all_simple_paths(G, start_id, goal_id):
        c = 0.0
        for a, b in zip(p, p[1:]):
            c += G[a][b]["weight"]
        min_cost = min(min_cost, c)
    assert abs(cost - min_cost) < 1e-6


def test_stop_background_tasks_idempotent() -> None:
    """Background maintenance thread can be stopped multiple times."""

    g = PlaceGraph(config={"decay_rate": 0.1})
    g.start_background_tasks(interval=0.01)
    g.observe("a")
    g.observe("b")
    time.sleep(0.02)
    g.stop_background_tasks()
    g.stop_background_tasks()
