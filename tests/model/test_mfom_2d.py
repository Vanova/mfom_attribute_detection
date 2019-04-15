"""
    Simulate multi-label classification.
"""
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Input
import keras.regularizers as regs
import keras.constraints as constraints
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import src.model.mfom as mfom
import src.utils.metrics as MT
import src.model.objectives as obj

RANDOM_SEED = 777
np.random.seed(RANDOM_SEED)


def generate_dataset(n_smp=300, ratio=0.3, n_feat=2, n_cls=2):
    x, y = make_multilabel_classification(n_samples=n_smp, n_features=n_feat,
                                          n_classes=n_cls, n_labels=1,
                                          allow_unlabeled=False,
                                          random_state=RANDOM_SEED)
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)
    x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, test_size=ratio, random_state=RANDOM_SEED)
    return x_tr, x_tst, y_tr, y_tst


def mfom_model(in_dim, nclass):
    # input block
    feat_input = Input(shape=(in_dim,), name='main_input')
    # layer 1
    x = Dense(10, name='dense1')(feat_input)
    x = Activation(activation='sigmoid', name='act1')(x)
    # layer 2
    x = Dense(10, name='dense2')(x)
    x = Activation(activation='sigmoid', name='act2')(x)
    # output layer
    x = Dense(nclass, name='pre_activation')(x)
    y_pred = Activation(activation='sigmoid', name='output')(x)

    # === MFoM head ===
    # misclassification layer, feed Y
    y_true = Input(shape=(nclass,), name='y_true')
    psi = mfom.UvZMisclassification(name='uvz_misclass')([y_true, y_pred])

    # class Loss function layer
    # NOTE: you may want to add regularization or constraints
    out = mfom.SmoothErrorCounter(name='smooth_error_counter',
                                  # alpha_constraint=constraints.min_max_norm(min_value=-4., max_value=4.),
                                  # alpha_regularizer=regs.l1(0.001),
                                  # beta_constraint=constraints.min_max_norm(min_value=-4., max_value=4.),
                                  # beta_regularizer=regs.l1(0.001)
                                  )(psi)

    # compile model
    model = Model(input=[y_true, feat_input], output=out)
    return model


def cut_mfom(model):
    # calc accuracy: cut MFoM head, up to sigmoid output
    input = model.get_layer(name='main_input').output
    out = model.get_layer(name='output').output
    cut_net = Model(input=input, output=out)
    return cut_net


if __name__ == '__main__':
    dim = 20
    nclass = 10

    # mfom model
    model = mfom_model(dim, nclass)
    model.compile(loss=obj.mfom_eer_normalized, optimizer='Adam')
    model.summary()

    # training on multi-label dataset
    x_train, x_test, y_train, y_test = generate_dataset(n_smp=10000, n_feat=dim, n_cls=nclass)
    mask = y_train.sum(axis=-1) != nclass
    y_train = y_train[mask]
    x_train = x_train[mask]
    hist = model.fit([y_train, x_train], y_train, nb_epoch=10, batch_size=16)

    # cut MFoM head
    cut_model = cut_mfom(model)
    y_pred = cut_model.predict(x_test)

    # evaluate
    eer_val = MT.eer(y_true=y_test.flatten(), y_pred=y_pred.flatten())
    print('EER: %.4f' % eer_val)

    # history plot, alpha and beta params of MFoM
    m = model.get_layer('smooth_error_counter')
    print('alpha: ', K.get_value(m.alpha))
    print('beta: ', K.get_value(m.beta))
    plt.plot(hist.history['loss'])
    plt.show()
