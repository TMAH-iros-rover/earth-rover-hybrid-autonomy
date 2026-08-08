from research.training.datasets.action_labels import action_to_linear_angular, classify_action


def test_classify_stop():
    assert classify_action(0.0, 0.0) == "STOP"


def test_classify_reverse_has_priority():
    assert classify_action(-0.2, 0.6) == "REVERSE"


def test_classify_left_right_forward():
    assert classify_action(0.2, 0.4) == "LEFT"
    assert classify_action(0.2, -0.4) == "RIGHT"
    assert classify_action(0.2, 0.0) == "FORWARD"


def test_action_to_linear_angular_accepts_dict_and_sequence():
    assert action_to_linear_angular({"linear": 0.1, "angular": -0.2}) == (0.1, -0.2)
    assert action_to_linear_angular([0.3, 0.4]) == (0.3, 0.4)
