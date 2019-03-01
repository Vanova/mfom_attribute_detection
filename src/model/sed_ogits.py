import keras.backend as K
from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Conv2D, Activation, \
    Dropout, Reshape, Input, BatchNormalization, GRU, Bidirectional, Permute, TimeDistributed
from keras.optimizers import Adam, SGD, Adadelta, RMSprop
import src.model.mfom as mfom
import src.model.objectives as obj
from src.base.model import BaseModel


class SEDOgitsModel(BaseModel):
    """
    The Sound Event Detection model.
    It has time distributed output layer
    # Arguments
        input shape: [batch_sz; band; frame_wnd; channel]
    """

    def __init__(self, config, input_shape, nclass):
        super(SEDOgitsModel, self).__init__(config)
        self.input_shape = input_shape
        self.nclass = nclass
        self.build()

    def build(self):
        """
        Construct the main structure of the network
        """
        print('DNN input shape', self.input_shape)

        if K.image_dim_ordering() == 'tf':
            batch_sz, bands, frames, channels = self.input_shape
            assert channels >= 1
            channel_axis = 3
            freq_axis = 1
            nn_shape = (bands, frames, channels)
        else:
            raise NotImplementedError('[ERROR] Only for TensorFlow background.')

        nb_filters = self.config['feature_maps']
        dropout_rate = self.config['dropout']
        pool_sz = [5, 2, 2]  # max-pooling across frequency only
        gru_nb = [32]  # [32, 32]
        fc_nb = [32]

        # Input block
        feat_input = Input(shape=nn_shape, name='input')
        x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(feat_input)
        # CNN block
        for sz in pool_sz:
            x = Conv2D(filters=nb_filters, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization(axis=channel_axis)(x)
            x = Activation(self.config['activation'])(x)
            x = MaxPooling2D(pool_size=(sz, 1))(x)
            x = Dropout(dropout_rate)(x)
        x = Permute((2, 1, 3))(x)
        x = Reshape((frames, -1))(x)
        # GRU block
        for n in gru_nb:
            x = Bidirectional(
                GRU(n, activation='tanh', dropout=dropout_rate,
                    recurrent_dropout=dropout_rate, return_sequences=True),
                merge_mode='mul')(x)
        # Fully connected
        for n in fc_nb:
            x = TimeDistributed(Dense(n))(x)
            x = Dropout(dropout_rate)(x)
        x = TimeDistributed(Dense(self.nclass))(x)

        # out dim: [batch, frames, nclass]
        y_pred = Activation(activation=self.config['out_score'], name='output')(x)
        self._compile_model(input=feat_input, output=y_pred, params=self.config)

    def rebuild(self, new_config):
        """
        Recompile the model with the new hyper parameters.
        NOTE: network topology is changing according to the 'new_config'
        """
        self.config.update(new_config)
        batch_sz, bands, frames, channels = self.input_shape
        self.input_shape = (self.config['batch'], bands, self.config['context_wnd'], channels)
        self.build()

    def chage_optimizer(self, new_config, change_out_unit=False):
        """
        Recompile the model with the new loss and optimizer.
        NOTE: network topology is not changing.
        """
        if new_config['freeze_wt']:
            # train only the top layers,
            # i.e. freeze all lower layers
            for layer in self.model.layers[:-4]:
                layer.trainable = False

        # cut MFoM layers: use only output prediction scores
        input = self.model.get_layer(name='input').output
        output = self.model.get_layer(name='output').output

        if change_out_unit:
            la = self.model.layers[-2].output
            output = Activation(activation=new_config['out_score'], name='output')(la)
            print('[INFO] output scores has been changed: %s to %s' % (self.config['out_score'], new_config['out_score']))

        self._compile_model(input=input, output=output, params=new_config)

    def forward(self, x):
        out_model = self.model
        if self.model.loss in obj.MFOM_OBJECTIVES:
            input = self.model.get_layer(name='input').output
            preact = self.model.get_layer(name='output').output
            out_model = Model(input=input, output=preact)
        return out_model.predict(x)

    def _compile_model(self, input, output, params):
        """
        Compile network structure with particular loss and optimizer
        """
        # ===
        # choose loss
        # ===
        if params['loss'] in obj.MFOM_OBJECTIVES:
            # add 2 layers for Maximal Figure-of-Merit
            _, _, frames, _ = self.input_shape
            y_true = Input(shape=(frames, self.nclass), name='y_true')
            psi = mfom.UvZMisclassification(name='uvz_misclass')([y_true, output])
            y_pred = mfom.SmoothErrorCounter(name='smooth_error_counter')(psi)

            # MFoM need labels info during training
            input = [y_true, input]
            output = y_pred
            loss = obj.MFOM_OBJECTIVES[params['loss']]
        elif params['loss'] == obj.mfom_eer_embed.__name__:
            loss = obj.mfom_eer_embed
        else:
            loss = params['loss']
        # ===
        # choose optimizer
        # ===
        if params['optimizer'] == 'adam':
            optimizer = Adam(lr=params['learn_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif params['optimizer'] == 'sgd':
            optimizer = SGD(lr=params['learn_rate'], decay=1e-6, momentum=0.9, nesterov=True)
        elif params['optimizer'] == 'adadelta':
            optimizer = Adadelta(lr=params['learn_rate'])
        elif params['optimizer'] == 'rmsprop':
            optimizer = RMSprop(lr=params['learn_rate'])
        else:
            optimizer = params['optimizer']

        self.model = Model(input=input, output=output)
        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()
