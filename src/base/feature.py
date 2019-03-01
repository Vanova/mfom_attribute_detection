import scipy.signal as scp_sig
import numpy as np


class BaseFeatureExtractor(object):
    EPS = np.spacing(1)

    def __init__(self, *args, **kwargs):
        self.windows = dict(hamming_asymmetric=lambda sz: scp_sig.hamming(sz, sym=False),
                            hamming_symmetric=lambda sz: scp_sig.hamming(sz, sym=True),
                            hann_asymmetric=lambda sz: scp_sig.hann(sz, sym=False),
                            hann_symmetric=lambda sz: scp_sig.hann(sz, sym=True))

    def feat_dim(self):
        raise NotImplementedError

    def extract(self, x, sample_rate):
        raise NotImplementedError

    def _window(self, wtype, smp_size):
        return self.windows[wtype](smp_size)

    def _subsequence(self, x, wnd):
        """
        Chunk the sequence x with 50% overlapping
        x: ndarray.
        wnd: Integer.
        """
        for index in xrange(0, len(x) - wnd + 1, wnd // 2):
            yield x[index:index + wnd]
