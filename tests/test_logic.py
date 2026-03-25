from rag_service.logic import should_reject


def test_should_reject_when_scores_below_thresholds():
    assert should_reject({"faithfulness": 0.2, "relevance": 0.9}) is True
    assert should_reject({"faithfulness": 0.9, "relevance": 0.2}) is True


def test_should_not_reject_when_scores_valid_and_high():
    assert should_reject({"faithfulness": 0.95, "relevance": 0.88}) is False


def test_should_not_reject_when_scores_malformed():
    assert should_reject({"faithfulness": "n/a", "relevance": 0.9}) is False
    assert should_reject({"faithfulness": None, "relevance": None}) is False
