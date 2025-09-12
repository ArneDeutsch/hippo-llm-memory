# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from hippo_eval.tasks.spatial.generator_multi import generate_spatial_multi


def test_generate_spatial_multi_basic() -> None:
    data = generate_spatial_multi(num_teach=2, num_test=1, seed=0)
    assert "teach" in data and "test" in data
    assert len(data["teach"]) == 2
    ep = data["teach"][0]
    assert ep["context_key"] == data["topology_id"]
    assert "episode_id" in ep
