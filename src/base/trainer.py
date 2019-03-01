class BaseTrainer(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data  # DataLoader
        self.config = config

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class BaseModelValidator(object):
    @staticmethod
    def validate_model(model, data, metrics):
        raise NotImplementedError


class BaseScoreDecision(object):
    def __init__(self, decision, threshold=0.5):
        self.decision = decision
        self.threshold = threshold

    def decide(self, y_pred, y_true=None):
        raise NotImplementedError

