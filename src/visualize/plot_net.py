import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Conv2D, Activation, \
    InputLayer, BatchNormalization, GRU, Bidirectional, TimeDistributed
import numpy as np


class NetworkVisualizer(object):
    """
    Keras networks visualizer
    """
    MAX_CHN_PLOT = 256

    def __init__(self, model):
        self.net = model

    def plot_weights(self, layer, file_name, show=True):
        w = layer.get_weights()
        if isinstance(layer, Conv2D):
            self.cnn_plot(w, file_name, show)
        elif isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            self.gru_plot(w, file_name, show)
        elif isinstance(layer, Dense) or isinstance(layer, TimeDistributed):
            self.dense_plot(w, file_name, show)
        else:
            print('[warn] plotter for layer %s is not implemented...' % layer)

    def gru_plot(self, weights, file_name, show):
        print('Plot GRU')
        nw = len(weights)
        if nw == 6:
            for i, (a, b) in enumerate(zip(weights[:2], weights[3:])):
                m = np.stack((a, b), axis=2)
                mosaic = self._tile_images(m, 1, 2)

                self._plot(file_name, mosaic, show)

    def dense_plot(self, weights, file_name, show):
        print('Plot Dense')
        if len(weights) == 2:
            weights = weights[0]  # take main weights, skip bias

        self._plot(file_name, weights, show)

    def cnn_plot(self, weights, file_name, show):
        if len(weights) == 2:
            weights = weights[0]  # take main weights, skip bias

        weights = weights.reshape((weights.shape[0], weights.shape[1], -1))
        weights = weights[:, :, :self.MAX_CHN_PLOT]
        print('W shape: ', weights.shape)

        s = int(np.sqrt(weights.shape[-1]) + 1)
        mosaic = self._tile_images(weights, s, s)

        self._plot(file_name, mosaic, show)

    def _tile_images(self, imgs, nrows, ncols, border=1):
        """
        Given a set of images with all the same shape, makes a
        mosaic with nrows and ncols
        Arguments
            imgs: ndarray, [width, height, nums]
        """
        w, h, sz = imgs.shape
        tile = np.ma.masked_all((nrows * w + (nrows - 1) * border,
                                 ncols * h + (ncols - 1) * border),
                                dtype=np.float32)
        paddedh = w + border
        paddedw = h + border
        for i in xrange(sz):
            row = int(np.floor(i / ncols))
            col = i % ncols
            tile[row * paddedh:row * paddedh + w,
            col * paddedw:col * paddedw + h] = imgs[:, :, i]
        return tile

    def _nice_imshow(self, ax, data, vmin=None, vmax=None, cmap=None):
        """Wrapper around pl.imshow"""
        if cmap is None:
            # cmap = cm.jet
            cmap = 'viridis'
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
        return im

    def _plot(self, file_name, mosaic, show):
        plt.figure(figsize=(15, 15))
        plt.title(file_name)
        self._nice_imshow(plt.gca(), mosaic)
        plt.axis('off')
        plt.savefig(file_name, dpi=400)
        if show:
            plt.show()

    def plot_activations(self, data, layer, file_name, rnd_channel, show):
        """
        channel: choose either 'random' or 'all'
        """
        lplot = self.net.get_layer(name=layer.name).output
        # creates a model to forward
        nn_forward = Model(input=self.net.input, output=lplot)
        acts = nn_forward.predict_on_batch(data)
        print(layer.name, acts.shape)

        smp_id = 30  # np.random.randint(0, acts.shape[0])
        if len(acts.shape) == 4:
            # CNN or Pooling: [smp, w, h, ch]
            ch_id = np.random.randint(0, acts.shape[-1])
            arr = acts[smp_id, :, :, ch_id]
        elif len(acts.shape) == 3:
            # e.g. for GRU: [smp, time, out_dim]
            arr = acts[smp_id, :, :]

        self._plot(file_name, arr, show)
