import numpy as np


def conformal_scores_from_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute split conformal scores for classification:
        s_i = 1 - p_{true_label}(x_i)

    Parameters
    ----------
    probs : np.ndarray
        Array of shape (n, K), predicted probabilities.
    labels : np.ndarray
        Array of shape (n,), true class labels.

    Returns
    -------
    np.ndarray
        Array of shape (n,), conformal scores.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    n = labels.shape[0]
    true_probs = probs[np.arange(n), labels]
    scores = 1.0 - true_probs
    return scores


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    """
    Compute split conformal threshold:
        qhat = Quantile_{ceil((n+1)(1-alpha))/n}(scores)

    Parameters
    ----------
    scores : np.ndarray
        Calibration scores of shape (n,).
    alpha : float
        Miscoverage level. Example: alpha=0.1 => target coverage 90%.

    Returns
    -------
    float
        Conformal threshold qhat.
    """
    scores = np.asarray(scores, dtype=np.float64)

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    n = scores.shape[0]
    if n == 0:
        raise ValueError("scores must be non-empty.")

    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(q_level, 1.0)

    try:
        qhat = np.quantile(scores, q_level, method="higher")
    except TypeError:
        # fallback for older numpy
        qhat = np.quantile(scores, q_level, interpolation="higher")

    return float(qhat)


def conformal_predict_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """
    Build prediction sets:
        C(x) = { y : 1 - p_y(x) <= qhat }

    Parameters
    ----------
    probs : np.ndarray
        Array of shape (m, K), predicted probabilities.
    qhat : float
        Conformal threshold.

    Returns
    -------
    np.ndarray
        Boolean array of shape (m, K).
        True means class is included in the prediction set.
    """
    probs = np.asarray(probs, dtype=np.float64)
    pred_sets = (1.0 - probs) <= qhat
    return pred_sets


def conformal_metrics(pred_sets: np.ndarray, labels: np.ndarray) -> dict:
    """
    Evaluate conformal prediction sets.

    Parameters
    ----------
    pred_sets : np.ndarray
        Boolean array of shape (m, K).
    labels : np.ndarray
        True labels of shape (m,).

    Returns
    -------
    dict
        coverage, avg_set_size, singleton_rate, empty_set_rate, class_wise_coverage
    """
    pred_sets = np.asarray(pred_sets, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)

    m = labels.shape[0]
    if m == 0:
        raise ValueError("labels must be non-empty.")

    covered = pred_sets[np.arange(m), labels]
    set_sizes = pred_sets.sum(axis=1)

    coverage = float(np.mean(covered))
    avg_set_size = float(np.mean(set_sizes))
    singleton_rate = float(np.mean(set_sizes == 1))
    empty_set_rate = float(np.mean(set_sizes == 0))

    class_wise_coverage = {}
    for c in np.unique(labels):
        idx = labels == c
        class_wise_coverage[int(c)] = float(np.mean(pred_sets[idx, c]))

    return {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "singleton_rate": singleton_rate,
        "empty_set_rate": empty_set_rate,
        "class_wise_coverage": class_wise_coverage,
    }


def run_split_conformal(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.1,
) -> dict:
    """
    End-to-end split conformal pipeline for classification.

    Returns a dictionary with:
      - alpha
      - qhat
      - calibration_scores
      - pred_sets
      - metrics
    """
    cal_scores = conformal_scores_from_probs(cal_probs, cal_labels)
    qhat = conformal_qhat(cal_scores, alpha)
    pred_sets = conformal_predict_sets(test_probs, qhat)
    metrics = conformal_metrics(pred_sets, test_labels)

    return {
        "alpha": alpha,
        "qhat": qhat,
        "calibration_scores": cal_scores,
        "pred_sets": pred_sets,
        "metrics": metrics,
    }




import numpy as np


def class_conditional_scores_from_probs(probs: np.ndarray, labels: np.ndarray, num_classes: int):
    """
    Returns a dict:
        class_id -> scores for calibration samples belonging to that class
    score_i = 1 - p_{true_label}(x_i)
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    scores_by_class = {}

    for c in range(num_classes):
        idx = labels == c
        class_probs = probs[idx, c]
        class_scores = 1.0 - class_probs
        scores_by_class[c] = class_scores

    return scores_by_class


def class_conditional_qhats(scores_by_class: dict, alpha: float):
    """
    Compute one qhat per class.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    qhats = {}

    for c, scores in scores_by_class.items():
        scores = np.asarray(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError(f"No calibration samples found for class {c}.")

        q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
        q_level = min(q_level, 1.0)

        try:
            qhat = np.quantile(scores, q_level, method="higher")
        except TypeError:
            qhat = np.quantile(scores, q_level, interpolation="higher")

        qhats[c] = float(qhat)

    return qhats


def class_conditional_predict_sets(probs: np.ndarray, qhats: dict):
    """
    probs: shape (m, K)
    qhats: dict class_id -> qhat_c

    returns:
        pred_sets: bool array (m, K)
    """
    probs = np.asarray(probs, dtype=np.float64)
    m, k = probs.shape

    pred_sets = np.zeros((m, k), dtype=bool)

    for c in range(k):
        pred_sets[:, c] = (1.0 - probs[:, c]) <= qhats[c]

    return pred_sets


def conformal_metrics(pred_sets: np.ndarray, labels: np.ndarray) -> dict:
    pred_sets = np.asarray(pred_sets, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)

    m = labels.shape[0]

    covered = pred_sets[np.arange(m), labels]
    set_sizes = pred_sets.sum(axis=1)

    coverage = float(np.mean(covered))
    avg_set_size = float(np.mean(set_sizes))
    singleton_rate = float(np.mean(set_sizes == 1))
    empty_set_rate = float(np.mean(set_sizes == 0))

    class_wise_coverage = {}
    for c in np.unique(labels):
        idx = labels == c
        class_wise_coverage[int(c)] = float(np.mean(pred_sets[idx, c]))

    return {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "singleton_rate": singleton_rate,
        "empty_set_rate": empty_set_rate,
        "class_wise_coverage": class_wise_coverage,
    }


def run_class_conditional_conformal(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.1,
):
    num_classes = cal_probs.shape[1]

    scores_by_class = class_conditional_scores_from_probs(
        cal_probs, cal_labels, num_classes=num_classes
    )
    qhats = class_conditional_qhats(scores_by_class, alpha=alpha)
    pred_sets = class_conditional_predict_sets(test_probs, qhats)
    metrics = conformal_metrics(pred_sets, test_labels)

    return {
        "alpha": alpha,
        "qhats": qhats,
        "scores_by_class": scores_by_class,
        "pred_sets": pred_sets,
        "metrics": metrics,
    }