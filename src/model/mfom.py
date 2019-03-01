"""
The Maximal Figure-of-Merit training divided in two layers.
 These are UvZMisclassification and SmoothErrorCounter.

 During the training stage combine your network with MFoM:
    DNN + UvZMisclassification + SmoothErrorCounter, we feed labels information into the MFoM;
    as the objective function use one of 'mfom_microf1', 'mfom_eer_normalized' or 'mfom_cprime'
 During the testing stage cut of the MFoM layers (UvZMisclassification and SmoothErrorCounter):
    because we don not have label information during testing.

# Citation
    "Deep learning with Maximal Figure-of-Merit Cost to Advance Multi-label Speech Attribute Detection"
        I. Kukanov, V. Hautam{\"a}ki, S. Siniscalchi, K. Li
    "Maximal Figure-of-Merit Embedding for Multi-label Audio Classification"
        I. Kukanov, V. Hautam{\"a}ki, K.A. Lee.
"""
from keras import backend as K
from keras.layers import Layer
from keras.layers.merge import _Merge
from keras import initializers
from keras import constraints
from keras import regularizers


class UvZMisclassification(_Merge):
    """
    Units-vs-zeros misclassification measure is the 1st-stage of the MFoM framework
    UvZMisclassification()([y_true, y_pred])

    # Attributes:
        input: list of two arrays
            y_true: labels needed only during training, [batch_sz, nclasses]
            y_pred: activations from the last layer, [batch_sz, nclasses]
    # Output:
    """
    _EPSILON = K.epsilon()

    def _merge_function(self, inputs):
        y_true = inputs[0]
        y_pred = inputs[1]
        out = self._uvz_misclass(y_true, y_pred)
        return out

    def _uvz_misclass(self, y_true, y_pred):
        y_neg = 1 - y_true
        # Kolmogorov log average of 'unit' labeled models
        unit_avg = y_true * K.exp(y_pred)
        # average over non-zero elements
        unit_avg = K.log(self._non_zero_mean(unit_avg))
        # Kolmogorov log average of 'zero' labeled models
        zeros_avg = y_neg * K.exp(y_pred)
        # average over non-zero elements
        zeros_avg = K.log(self._non_zero_mean(zeros_avg))
        # misclassification measure
        # TODO NOTE: sometimes works better with 0.5
        # psi = -y_pred + 0.5
        psi = -y_pred + y_neg * unit_avg + y_true * zeros_avg
        return psi

    def _non_zero_mean(self, x):
        # Average values which meet the criterion > 0
        mask = K.greater(K.abs(x), self._EPSILON)
        n = K.sum(K.cast(mask, 'float32'), axis=-1, keepdims=True)
        return K.sum(x, axis=-1, keepdims=True) / n


class SmoothErrorCounter(Layer):
    """
    This is the 2nd-stage of the Maximal Figure-of-Merit.
    Optimize alpha_k and beta_k - parameters of the Smooth Error function from MFoM.
    l_k = K.sigmoid(alpha_k * d + beta_k)

    see BatchNorm  for optimisation (gamma and beta)
    """

    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 alpha_init='one',
                 beta_init='zero',
                 alpha_regularizer=None,
                 beta_regularizer=None,
                 alpha_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        super(SmoothErrorCounter, self).__init__(**kwargs)
        # self.supports_masking = True
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.alpha_initializer = initializers.get(alpha_init)
        self.beta_initializer = initializers.get(beta_init)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.beta_constraint = constraints.get(beta_constraint)

    def build(self, input_shape):
        self.axis = -1
        dim = input_shape[self.axis]  # [batch_sz, nclasses] or [batch_sz, frames, nclasses]
        if dim is None:
            raise ValueError('Axis 1 of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        shape = (dim,)  # (nclasses,)

        if self.scale:
            self.alpha = self.add_weight(shape=shape,
                                         name='alpha',
                                         initializer=self.alpha_initializer,
                                         regularizer=self.alpha_regularizer,
                                         constraint=self.alpha_constraint)
        else:
            self.alpha = None

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        self.built = True

    def call(self, inputs, training=None):
        normed_training = K.sigmoid(inputs * self.alpha + self.beta)
        return normed_training
