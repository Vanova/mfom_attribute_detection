import numpy as np
import sklearn.metrics as skm


def eer(y_true, y_pred):
    """
    EER for binary classifier
        y_true: ndarray, [smps; 1]
        y_score: ndarray, [smps; 1], real valued scores of classifier
    # Output
        EER value
    """
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, drop_intermediate=True)

    eps = 1E-6
    points = [(0, 0)] + zip(fpr, tpr)
    for i, point in enumerate(points):
        if point[0] + eps >= 1 - point[1]:
            break
    p1 = points[i - 1]
    p2 = points[i]
    # Interpolate between p1 and p2
    if abs(p2[0] - p1[0]) < eps:
        res = p1[0]
    else:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        x = p1[1] - m * p1[0]
        res = (1 - x) / (1 + m)
    return 100. * res


def class_wise_eer(y_true, y_pred):
    """
    Calculate eer per each class, multi-class classifier
        Y_true: ndarray, [smps; n_class]
        Y_pred: ndarray, [smps; n_class]
    # Output
        list of eer values per class, n_class
    """
    cw_eer = []
    smp, n_clc = y_true.shape
    for cl in xrange(n_clc):
        er = eer(y_true=y_true[:, cl], y_pred=y_pred[:, cl])
        cw_eer.append(er)
    return cw_eer


def micro_f1(y_true, y_pred, accuracy=True):
    """
    Calculate micro-F1 measure for multi-class classifier
        y_true: ndarray, [smps; n_class]
        y_pred: ndarray, [smps; n_class], thresholded (with step function) binary integers
    # Output
        Accuracy or Error of micro-F1
    """
    assert (len(y_true) == len(y_pred))
    neg_r = np.logical_not(y_true)
    neg_p = np.logical_not(y_pred)
    tp = np.sum(np.logical_and(y_true, y_pred) == True)
    fp = np.sum(np.logical_and(neg_r, y_pred) == True)
    fn = np.sum(np.logical_and(y_true, neg_p) == True)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) * 100.
    return f1 if accuracy else 100. - f1


def pooled_accuracy(y_true, y_pred):
    """
    Accuracy for multi-class classifier,
        all scores are pooled in single list
        y_true: list, class ids
        y_pred: list, class ids
    # Output
        Accuracy
    """
    N = float(len(y_true))
    return sum(int(x == y) for (x, y) in zip(y_true, y_pred)) / N * 100.


def step(a, threshold=0.5):
    """
    Heaviside step function:
        a < threshold = 0, else 1.
        a: ndarray, [smps; n_class]
    # Output
        binary ndarray [smps; n_class]
    """
    res = np.zeros_like(a)
    res[a < threshold] = 0
    res[a >= threshold] = 1
    return res
