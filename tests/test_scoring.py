from hippo_eval.metrics.scoring import em_norm, em_raw


def test_em_norm_zero_when_raw_zero() -> None:
    pred = "The answer."
    gold = "answer"
    assert em_raw(pred, gold) == 0
    assert em_norm(pred, gold) == 0


def test_em_norm_matches_raw_when_exact() -> None:
    pred = "answer"
    gold = "answer"
    assert em_raw(pred, gold) == 1
    assert em_norm(pred, gold) == 1


def test_em_norm_strips_non_moves() -> None:
    pred = "U, D, L, R"
    gold = "U D L R"
    assert em_norm(pred, gold) == 1
