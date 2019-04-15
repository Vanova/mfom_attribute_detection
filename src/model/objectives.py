"""
# Citations
    "Deep learning with Maximal Figure-of-Merit Cost to Advance Multi-label Speech Attribute Detection"
        I. Kukanov, V. Hautam{\"a}ki, M. Siniscalchi, K. Li.
    "Maximal Figure-of-Merit Embedding for Multi-label Audio Classification"
        I. Kukanov, V. Hautam{\"a}ki, K.A. Lee.
"""
import keras.backend as K

_EPSILON = K.epsilon()


def mfom_eer_normalized(y_true, y_pred):
    """
    Class-wise MFoM-EER, i.e., macro-averaging EER approximation
    NOTE: it is meant to work with 'UvZMisclassification' and 'SmoothErrorCounter' layers.
    Here y_pred is L_k smooth error function
    # Arguments
        y_true: tensor, true targets, shape: [smp_size, dim] or [smp_size, frames, dim]
        y_pred: tensor, predicted errors, shape: [smp_size, dim] or [smp_size, frames, dim]
    """
    s = K.shape(y_true)
    y_true = K.reshape(y_true, (-1, s[-1]))
    y_pred = K.reshape(y_pred, (-1, s[-1]))
    y_neg = 1 - y_true
    # number of positive samples per each class
    P = K.sum(y_true, axis=0)
    # number of negative samples per each class
    N = K.sum(y_neg, axis=0)
    # smooth false negative and false positive
    fn = y_pred * y_true
    fp = (1. - y_pred) * y_neg
    # smooth false negative and false positive rates
    fnr = K.log(K.sum(fn, axis=0) + 1.) - K.log(P + 1.)
    fpr = K.log(K.sum(fp, axis=0) + 1.) - K.log(N + 1.)
    # debug output
    # fnr = K.print_tensor(K.exp(fnr), message="FNR is: ")
    # fpr = K.print_tensor(K.exp(fpr), message="FPR is: ")
    fnr = K.exp(fnr)
    fpr = K.exp(fpr)
    smooth_eer = fpr + .5 * K.abs(fnr - fpr)
    return K.mean(smooth_eer)


def pooled_mfom_eer(y_true, y_pred):
    """
    Pooled MFoM-EER, i.e., micro-averaging EER approximation
    NOTE: it is supposed to work with 'UvZMisclassification' and 'SmoothErrorCounter' layers.
    Here y_pred is L_k smooth error function
    # Arguments
        y_true: tensor, true targets, shape: [smp_size, dim] or [smp_size, frames, dim]
        y_pred: tensor, predicted errors, shape: [smp_size, dim] or [smp_size, frames, dim]
    """
    y_neg = 1 - y_true
    # number of positive samples
    P = K.sum(y_true)
    # number of negative samples
    N = K.sum(y_neg)
    # smooth false negative and false positive
    fn = y_pred * y_true
    fp = (1. - y_pred) * y_neg
    # smooth false negative and false positive rates
    fnr = K.sum(fn) / P
    fpr = K.sum(fp) / N
    smooth_eer = fpr + .5 * K.abs(fnr - fpr)
    return smooth_eer


def mfom_microf1(y_true, y_pred):
    """
    MFoM micro F1, i.e. micro-averaging F1 approximation (pool all scores and calculate errors)
    NOTE: it is supposed to work with 'UvZMisclassification' and 'SmoothErrorCounter' layers.
    Here y_pred is L_k smooth error function
    # Arguments
        y_true: tensor, true targets, shape: [smp_size, dim] or [smp_size, frames, dim]
        y_pred: tensor, predicted errors, shape: [smp_size, dim] or [smp_size, frames, dim]
    """
    p = 1. - y_pred
    numen = 2. * K.sum(p * y_true)
    denum = K.sum(p + y_true)
    smooth_f1 = numen / denum
    return 1.0 - smooth_f1


def mfom_macrof1(y_true, y_pred):
    """
    MFoM macro F1, i.e. micro-averaging F1 approximation (calculate errors per class)
    NOTE: it is supposed to work with 'UvZMisclassification' and 'SmoothErrorCounter' layers.
    Here y_pred is L_k smooth error function
    # Arguments
        y_true: tensor, true targets, shape: [smp_size, dim] or [smp_size, frames, dim]
        y_pred: tensor, predicted errors, shape: [smp_size, dim] or [smp_size, frames, dim]
    """
    s = K.shape(y_true)
    y_true = K.reshape(y_true, (-1, s[-1]))
    y_pred = K.reshape(y_pred, (-1, s[-1]))
    y_neg = 1 - y_true
    # smooth counters per class
    tp = K.sum((1. - y_pred) * y_true, axis=0)
    fn = K.sum(y_pred * y_true, axis=0)
    fp = K.sum((1. - y_pred) * y_neg, axis=0)
    numen = 2. * tp
    denum = fp + fn + 2. * tp
    smooth_f1 = K.exp(K.log(numen + 1.) - K.log(denum + 1.))
    error_f1 = 1. - K.mean(smooth_f1)
    # debug output
    # tp = K.print_tensor(tp, message='TP is: ')
    # fn = K.print_tensor(fn, message='FN is: ')
    # fp = K.print_tensor(fp, message='FP is: ')
    # error_f1 = K.print_tensor(error_f1, message='error_f1: ')
    return error_f1


def mfom_cprime(y_true, y_pred, ptar=0.01):
    """
    Objective function C_prime of NIST SRE/LRE challenges
    NOTE: it is supposed to work with 'UvZMisclassification' and 'SmoothErrorCounter' layers.
    Here y_pred is L_k smooth error function
    # Arguments
        y_true: tensor, true targets, shape: [smp_size, dim] or [smp_size, frames, dim]
        y_pred: tensor, predicted errors, shape: [smp_size, dim] or [smp_size, frames, dim]
    """
    y_neg = 1 - y_true
    # number of positive samples
    P = K.sum(y_true)
    # number of negative samples
    N = K.sum(y_neg)
    fn = y_pred * y_true
    fp = (1. - y_pred) * y_neg
    # === pooled
    fnr = K.sum(fn) / P
    fpr = K.sum(fp) / N
    smooth_eer = ptar * fnr + (1. - ptar) * fpr  # TODO adjust a constant
    return smooth_eer


def mfom_eer_embed(y_true, y_pred):
    """
    MFoM embedding: use MFoM scores as new "soft labels", a.k.a. Dark Knowledge by G. Hinton
    NOTE: can work without MFoM layers.
    # Arguments
        y_true: tensor, true targets, shape: [smp_size, dim] or [smp_size, frames, dim]
        y_pred: tensor, predicted errors, shape: [smp_size, dim] or [smp_size, frames, dim]
    """
    alpha = 3.  # 10., 5.0 # 0.1, 1., 10, 5.0, 3.0
    beta = 0.0
    n_embed = 2
    l = _uvz_loss_scores(y_true, y_pred, alpha, beta)
    l_score = 1 - l
    for t in xrange(n_embed):
        l = _uvz_loss_scores(y_true=y_true, y_pred=l_score, alpha=alpha, beta=beta)
        l_score = 1 - l
    # ===
    # MSE(y_pred - l_score)
    # ===
    # mse = K.mean(K.square(y_pred - l_score3), axis=-1)
    # ===
    # binXent(y_pred - l_score)
    # ===
    binxent = K.mean(K.binary_crossentropy(y_pred, l_score), axis=-1)
    # ===
    # Xent(y_pred - l_score)
    # ===
    # xent = K.categorical_crossentropy(y_pred, l_score)
    return binxent


def _uvz_loss_scores(y_true, y_pred, alpha, beta):
    y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    y_neg = 1 - y_true
    # Kolmogorov log average of unit labeled models
    unit_avg = y_true * K.exp(y_pred)
    # average over non-zero elements
    unit_avg = K.log(_non_zero_mean(unit_avg))
    # Kolmogorov log average of zero labeled models
    zeros_avg = y_neg * K.exp(y_pred)
    # average over non-zero elements
    zeros_avg = K.log(_non_zero_mean(zeros_avg))
    # misclassification measure, optimized
    # TODO sometimes also works
    # d = -y_pred + 0.5
    d = -y_pred + y_neg * unit_avg + y_true * zeros_avg
    # calculate class loss function l
    l = K.sigmoid(alpha * d + beta)
    return l


def _non_zero_mean(x):
    # All values which meet the criterion > 0
    mask = K.greater(K.abs(x), _EPSILON)
    n = K.sum(K.cast(mask, 'float32'), axis=-1, keepdims=True)
    return K.sum(x, axis=-1, keepdims=True) / n


MFOM_OBJECTIVES = dict(mfom_eer_normalized=mfom_eer_normalized,
                       pooled_mfom_eer=pooled_mfom_eer,
                       mfom_microf1=mfom_microf1,
                       mfom_macrof1=mfom_macrof1,
                       mfom_cprime=mfom_cprime)
