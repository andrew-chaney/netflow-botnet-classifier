from math import sqrt


def precision(tp: int, fp: int) -> float:
    """
    Calculates the precision value.

    :param tp: true positives
    :param fp: false positives
    :returns: the precision value
    """
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    """
    Calculates the recall (sensitivity) value.

    :param tp: true positives
    :param fn: false negatives
    :returns: the recall value
    """
    return tp / (tp + fn)


def f1(tp: int, fp: int, fn: int) -> float:
    """
    Calculates the F-1 Score.

    :param tp: true positives
    :param fp: false positives
    :param fn: false negatives
    :returns: the F-1 Score
    """
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return 2 * ((prec * rec) / (prec + rec))


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculates the Matthew's Correlation Coefficient (MCC).

    :param tp: true positives
    :param tn: true negatives
    :param fp: false positives
    :param fn: false negatives
    :returns: the MCC
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator
