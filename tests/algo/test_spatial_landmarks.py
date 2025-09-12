# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from hippo_mem.spatial.map import PlaceGraph


def test_landmark_write_and_counters() -> None:
    g = PlaceGraph()
    g.add_landmark("room", (2.0, -1.0), kind="room")
    g.add_landmark("door", (0.5, 0.5), kind="door")
    g.connect("room", "door", kind="door")

    status = g.log_status()
    assert status["landmarks_added"] == 2
    assert status["edges_added"] >= 1

    place = g.encoder.encode("room")
    assert place.coord == (1.0, 0.0)
    assert place.kind == "room"
    edge = g.graph[g._context_to_id["room"]][g._context_to_id["door"]]
    assert edge.kind == "door"
