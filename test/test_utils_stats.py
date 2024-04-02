import pytest

from src.utils.stats import f1, mcc, precision, recall


@pytest.mark.parametrize(
    "arg1, arg2, expected_result",
    [
        (10, 1, 0.9090909091),
        (40, 25, 0.6153846154),
        (80, 20, 0.8),
    ],
)
def test_precision(arg1, arg2, expected_result):
    precision_result = precision(arg1, arg2)
    assert round(precision_result * 100, 3) == round(expected_result * 100, 3)


@pytest.mark.parametrize(
    "arg1, arg2, expected_result",
    [
        (32, 5, 0.8648648649),
        (19, 52, 0.2676056338),
        (98, 10, 0.9074074074),
    ],
)
def test_recall(arg1, arg2, expected_result):
    recall_result = recall(arg1, arg2)
    assert round(recall_result * 100, 3) == round(expected_result * 100, 3)


@pytest.mark.parametrize(
    "arg1, arg2, arg3, expected_result",
    [
        (385, 493, 15, 0.6025039123219972),
        (504, 694, 151, 0.5439827307069616),
        (243, 232, 100, 0.5941320293398534),
    ],
)
def test_f1(arg1, arg2, arg3, expected_result):
    f1_result = f1(arg1, arg2, arg3)
    assert round(f1_result * 100, 3) == round(expected_result * 100, 3)


@pytest.mark.parametrize(
    "arg1, arg2, arg3, arg4, expected_result",
    [
        (104, 859, 999, 303, -0.21672),
        (649, 550, 506, 471, 0.10044),
        (663, 551, 184, 911, 0.16517),
    ],
)
def test_mcc(arg1, arg2, arg3, arg4, expected_result):
    mcc_result = mcc(arg1, arg2, arg3, arg4)
    assert round(mcc_result * 100, 3) == round(expected_result * 100, 3)
