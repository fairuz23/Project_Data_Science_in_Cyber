from src.evaluation import evaluate


def test_evaluate_perfect_predictions():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 1]
    metrics = evaluate(y_true, y_pred, label="perfect")
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["pipeline"] == "perfect"


def test_evaluate_flags_false_negative_over_false_positive_asymmetry():
    # one missed malware sample (false negative): true=1, pred=0
    y_true = [1, 1, 0, 0]
    y_pred = [0, 1, 0, 0]
    metrics = evaluate(y_true, y_pred, label="one_fn")
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0
    assert 0 < metrics["mcc"] < 1
