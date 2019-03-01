
class BasePipeline(object):
    """Main end-point to run the pipeline processes"""
    def __init__(self, config):
        self.config = config

    def initialize_data(self):
        """ Initialize dataset: loading, preprocess metadata"""
        raise NotImplementedError

    def extract_feature(self):
        raise NotImplementedError

    def normalize_feature(self):
        raise NotImplementedError

    def search_hyperparams(self):
        raise NotImplementedError

    def system_train(self):
        raise NotImplementedError

    def system_test(self):
        raise NotImplementedError

    def system_evaluate(self):
        raise NotImplementedError

    def show_parameters(self):
        raise NotImplementedError

    def show_dataset_info(self):
        raise NotImplementedError