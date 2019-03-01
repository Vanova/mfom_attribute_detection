import sys
import numpy as np
import numpy.random as rnd
from keras import backend as K
import src.model.objectives as obj
import src.utils.metrics as mtr

rnd.seed(123)
_EPSILON = K.epsilon()
allobj = [obj.mfom_eer_normalized,
          obj.pooled_mfom_eer,
          obj.mfom_microf1,
          obj.mfom_macrof1,
          obj.mfom_cprime]


def mfom_eer_normalized_np(y_true, y_pred):
    """
    Class-wise MFoM-EER numpy version
    """
    s = y_true.shape
    y_true = np.reshape(y_true, (-1, s[-1]))
    y_pred = np.reshape(y_pred, (-1, s[-1]))
    y_neg = 1 - y_true
    # number of positive samples per each class
    P = np.sum(y_true, axis=0)
    # number of negative samples per each class
    N = np.sum(y_neg, axis=0)
    # smooth false negative and false positive
    fn = y_pred * y_true
    fp = (1. - y_pred) * y_neg
    fnr = np.log(np.sum(fn, axis=0) + 1.) - np.log(P + 1.)
    fpr = np.log(np.sum(fp, axis=0) + 1.) - np.log(N + 1.)
    fnr = np.exp(fnr)
    fpr = np.exp(fpr)
    smooth_eer = fpr + .5 * np.abs(fnr - fpr)  # dim = number of classes
    return np.mean(smooth_eer)


def pooled_mfom_eer_np(y_true, y_pred):
    """
    Pooled MFoM-EER numpy version
    """
    y_neg = 1 - y_true
    # number of positive samples per each class
    P = np.sum(y_true)
    # number of negative samples per each class
    N = np.sum(y_neg)
    fn = y_pred * y_true
    fp = (1. - y_pred) * y_neg
    fnr = np.sum(fn) / P
    fpr = np.sum(fp) / N
    smooth_eer = fpr + .5 * np.abs(fnr - fpr)
    return smooth_eer


def mfom_microf1_np(y_true, y_pred):
    y_neg = 1 - y_true
    tp = np.sum((1. - y_pred) * y_true)
    fp = np.sum((1. - y_pred) * y_neg)
    fn = np.sum(y_pred * y_true)
    numen = 2. * tp
    denum = fp + fn + 2. * tp
    smooth_f1 = numen / denum
    return 1.0 - smooth_f1


def mfom_macrof1_np(y_true, y_pred):
    """
        Class-wise F1
    """
    s = y_true.shape
    y_true = np.reshape(y_true, (-1, s[-1]))
    y_pred = np.reshape(y_pred, (-1, s[-1]))
    y_neg = 1 - y_true
    # smooth counters per class
    tp = np.sum((1. - y_pred) * y_true, axis=0)
    fn = np.sum(y_pred * y_true, axis=0)
    fp = np.sum((1. - y_pred) * y_neg, axis=0)

    numen = 2. * tp
    denum = fp + fn + 2. * tp
    smooth_f1 = np.exp(np.log(numen + 1.) - np.log(denum + 1.))
    error_f1 = 1.0 - smooth_f1
    return np.mean(error_f1)


def mfom_cprime_np(y_true, y_pred, ptar=0.01):
    y_neg = 1 - y_true
    P = np.sum(y_true)
    N = np.sum(y_neg)
    fn = y_pred * y_true
    fp = (1. - y_pred) * y_neg
    # === pooled
    fnr = np.sum(fn) / P
    fpr = np.sum(fp) / N
    smooth_eer = ptar * fnr + (1. - ptar) * fpr
    return smooth_eer


def check_shape(shape, fun):
    y_true = rnd.choice([0, 1], size=shape, p=[2. / 3, 1. / 3])
    y_pred = rnd.uniform(0, 1, shape)
    fun_np = getattr(sys.modules[__name__], fun.__name__ + '_np')
    res_np = fun_np(y_true, y_pred)
    res_k = K.eval(fun(K.variable(y_true), K.variable(y_pred)))

    print(res_k, res_np)
    assert res_k.shape == res_np.shape
    assert np.isclose(res_k, res_np)
    print('pEER: %.3f' % mtr.eer(y_true.flatten(), y_pred.flatten()))


def test_objective_shapes():
    shape_list = [(6), (6, 7), (5, 6, 7), (8, 5, 6, 7), (9, 8, 5, 6, 7)]
    for sh in shape_list:
        for fun in allobj:
            print(fun.__name__)
            check_shape(sh, fun)
            print('=' * 10)


if __name__ == '__main__':
    test_objective_shapes()
