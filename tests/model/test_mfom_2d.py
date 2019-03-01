import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Input
import matplotlib.pyplot as plt
import src.model.mfom as mfom
import src.utils.metrics as MT
import src.model.objectives as obj

np.random.seed(777)


def generate_dataset(output_dim=14, num_examples=10000):
    """
    Summation of two binary numbers.
    Input is two binary numbers, stacked in one vector.
    Output is an integer number.
    """

    def int2vec(x, dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2 ** (output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2 ** (output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]), int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x)
    y = np.array(y)
    return x, y


if __name__ == '__main__':
    dim = 14
    nclass = 7

    # Input block
    feat_input = Input(shape=(dim,), name='main_input')
    # layer 1
    x = Dense(30, name='dense1')(feat_input)
    x = Activation(activation='sigmoid', name='act1')(x)
    # output layer
    x = Dense(nclass, name='pre_activation')(x)
    y_pred = Activation(activation='sigmoid', name='output')(x)

    # misclassification layer, feed Y
    y_true = Input(shape=(nclass,), name='y_true')
    psi = mfom.UvZMisclassification(name='uvz_misclass')([y_true, y_pred])

    # class Loss function layer
    out = mfom.SmoothErrorCounter(name='smooth_error_counter')(psi)

    # compile model
    model = Model(input=[y_true, feat_input], output=out)
    model.compile(loss=obj.mfom_eer_normalized, optimizer='Adam')
    model.summary()

    # train
    X, Y = generate_dataset(output_dim=nclass)
    hist = model.fit([Y, X], Y, nb_epoch=100, batch_size=16)

    # calc accuracy: we cut MFoM head, up to sigmoid output
    input = model.get_layer(name='main_input').output
    out = model.get_layer(name='output').output
    cut_model = Model(input=input, output=out)
    y_pred = cut_model.predict(X)
    eer_val = MT.eer(y_true=Y.flatten(), y_pred=y_pred.flatten())
    print('EER: %.4f' % eer_val)

    # history plot, alpha and beta params
    m = model.get_layer('smooth_error_counter')
    print('alpha: ', K.get_value(m.alpha))
    print('beta: ', K.get_value(m.beta))
    plt.plot(hist.history['loss'])
    plt.show()
