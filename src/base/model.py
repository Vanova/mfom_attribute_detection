"""
Base Model class
"""


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    def save(self, model_file):
        if self.model is None:
            raise Exception('You have to build the model first.')

        print('Saving model...')
        self.model.save_weights(model_file)
        print('Model saved')

    def load(self, model_file):
        if self.model is None:
            raise Exception('You have to build the model first.')

        print('Loading model checkpoint {} ...\n'.format(model_file))
        self.model.load_weights(model_file)
        print('Model loaded')

    def build(self):
        raise NotImplementedError

    def forward(self, data_loader):
        """
        Output depends on the application:
        e.g. for MNIST it is a score per image, for audio it is a score per frame
        """
        return self.model.predict(data_loader.eval_data())
