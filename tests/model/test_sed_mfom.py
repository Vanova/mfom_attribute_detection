import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Input, TimeDistributed, Dropout, Permute
import matplotlib.pyplot as plt
import src.model.mfom as mfom
import src.model.objectives as obj
import src.utils.metrics as MT
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
    nclass = 14

    # 3D input as fbank
    feat_dim, time_step = 28, 100
    feat_input = Input(shape=(feat_dim, time_step), name='main_input')
    x = Permute((2, 1))(feat_input)
    for _f in [256, 64]:
        x = TimeDistributed(Dense(_f))(x)
        # x = Activation(activation='elu')(x)
        x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(nclass))(x)
    y_pred = Activation(activation='tanh', name='output')(x)

    # misclassification layer, feed Y
    y_true = Input(shape=(time_step, nclass), name='y_true')
    psi = mfom.UvZMisclassification(name='uvz_misclass')([y_true, y_pred])

    # class Loss function layer
    out = mfom.SmoothErrorCounter(name='smooth_error_counter')(psi)
    # out = BatchNormalization()(psi)

    # compile model
    model = Model(input=[y_true, feat_input], output=out)
    model.compile(loss=obj.mfom_eer_normalized, optimizer='Adadelta') # Adam, Adadelta
    model.summary()

    # train
    all_X, all_Y = [], []
    for i in range(10000 // time_step):
        X, Y = generate_dataset(output_dim=nclass, num_examples=time_step)
        all_X.append(X.T)
        all_Y.append(Y)
    all_X, all_Y = np.array(all_X), np.array(all_Y)
    hist = model.fit([all_Y, all_X], all_Y, nb_epoch=200, batch_size=5)

    # calc EER: we cut MFoM head, up to sigmoid output
    input = model.get_layer(name='main_input').output
    out = model.get_layer(name='output').output
    cut_model = Model(input=input, output=out)
    y_pred_sig = cut_model.predict(all_X)
    eer_val = MT.eer(all_Y.flatten(), y_pred_sig.flatten())
    print('sigma_EER: %.4f' % eer_val)

    y_pred = model.predict([all_Y, all_X])
    eer_val = MT.eer(all_Y.flatten(), 1.-y_pred.flatten())
    print('l_EER: %.4f' % eer_val)
    print(model.evaluate([all_Y, all_X], all_Y))

    # TODO notice from the experiments:
    # when we minimize obj.mfom_microf1 with psi = y_pred or psi = -y_pred + 0.5 in
    # UvZMisclassification() layer, the smoothF1 is minimized !!! but EER is not at all.
    # When we minimize obj.mfom_microf1 with psi = -y_pred + y_neg * unit_avg + y_true * zeros_avg,
    # then both smoothF1 and EER are minimized :)

    # history plot, alpha and beta params
    m = model.get_layer('smooth_error_counter')
    print('alpha: ', K.get_value(m.alpha))
    print('beta: ', K.get_value(m.beta))

    # print stats of psi misclassification measure
    # m = model.get_layer('uvz_misclass')
    # print('stats_d: ', K.get_value(m.stats_psi))
    plt.plot(hist.history['loss'])
    plt.show()
