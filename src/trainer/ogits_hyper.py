import pprint
import pickle
import copy
import os.path as path
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from timeit import default_timer as timer
import src.utils.config as cfg
import src.data_loader.ogits as DL
from src.trainer.ogits import ModelValidator, OGITSModelTrainer

np.random.seed(777)


class HyperSearch(OGITSModelTrainer):
    """
    # Arguments
        model: BaseModel inherited, model for searching hyperparams
        data: BaseDataLoader, generate datasets (for train/test/validate)
        config: DotMap, initial configuration of the 'model',
                by default we choose 'pretrain_set', though these are changing
                during hyper search
        train_mode: String. 'pretrain_set' or 'finetune_set'.
        model: not trained or initially pre-trained model
    """
    MAX_RUNS = 1000
    MAX_EPOCHS = 20

    _search_space = {
        'out_score': hp.choice('out_score', ['tanh', 'sigmoid', 'relu']),
        'dropout': hp.choice('dropout', [0.1, 0.3, 0.5, 0.8]),
        'feature_maps': hp.choice('feature_maps', [16, 32, 64, 96, 128, 256]),
        'context_wnd': hp.choice('context_wnd', [16, 32, 64, 96, 128, 256]),
        'batch': hp.choice('batch', [8, 16, 32, 64, 128, 256]),
        'batch_type': hp.choice('batch_type', ['sed_random_crop', 'sed_sequence']),
        'learn_rate': hp.choice('learn_rate', [0.01, 0.001, 0.0001]),
        'activation': hp.choice('activation', ['relu', 'elu', 'sigmoid', 'tanh']),
        'loss': hp.choice('loss', ['mfom_eer_normalized',
                                   'mfom_microf1',
                                   'mfom_eer_embed',
                                   'mfom_cprim',
                                   'binary_crossentropy',
                                   'mse']),
        'optimizer': hp.choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adadelta'])
    }

    _search_space_tune = {
        'out_score': hp.choice('out_score',
                               ['tanh', 'sigmoid', 'exponential', 'linear',
                                'hard_sigmoid', 'softsign', 'softplus', 'selu']),
        'batch': hp.choice('batch', [8, 16, 32, 64, 128, 256]),
        'batch_type': hp.choice('batch_type', ['sed_random_crop', 'sed_sequence']),
        'learn_rate': hp.choice('learn_rate', [0.01, 0.001, 0.0001, 0.1, 1.]),
        'activation': hp.choice('activation', ['elu', 'sigmoid', 'tanh']),
        'loss': hp.choice('loss', ['mfom_eer_normalized',
                                   'mfom_microf1',
                                   'mfom_eer_embed',
                                   'mfom_cprime']),
        'optimizer': hp.choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adadelta'])
    }

    def __init__(self, model, data, config, **kwargs):
        super(HyperSearch, self).__init__(model, data, config, **kwargs)
        self.model_cfg = copy.deepcopy(self.config['model']['pretrain_set'])
        self.init_model_cfg = copy.deepcopy(self.config['model']['pretrain_set'])
        self.hp_srch_dir = self.config['path']['hyper_search']
        self.history_file = self.hp_srch_dir + '/history.tsv'
        self.trial_file = self.hp_srch_dir + '/trials.pkl'
        self.hp_history = pd.DataFrame()

    def train(self):
        """
        Run trial loop optimizer on search space
        # Return
            save the best value of metrics and hyperparameters
        """
        step = 2  # number of trials every run
        cumulative_trials = 5
        for i in xrange(self.MAX_RUNS // step):
            try:
                trials = pickle.load(open(self.trial_file, 'rb'))
                print('[INFO] Found saved Trials! Loading...')
                cumulative_trials = len(trials.trials) + step
                print('Rerunning from %d trial to %d (+%d) trials' % (len(trials.trials), cumulative_trials, step))
            except:
                trials = Trials()

            fmin(fn=self._experiment,
                 space=self._search_space_tune,
                 algo=tpe.suggest, max_evals=cumulative_trials, trials=trials)

            print('BEST hyperparams so far:')
            sort_res = sorted(trials.results, key=lambda x: x['loss'])
            pprint.pprint(sort_res[:1])

            # save the trials
            with open(self.trial_file, 'wb') as f:
                pickle.dump(trials, f)

            # save trial history
            with open(self.history_file, 'w') as f:
                self.hp_history = pd.DataFrame(sort_res)
                self.hp_history.reset_index(inplace=True, drop=True)
                self.hp_history.to_csv(f, sep='\t')

    def _experiment(self, params):
        """
        Calculate model validation error on the current hyper parameters 'params'.
        Hyper parameters are sampled from the search space.
        NOTE: we have to use only validation data set here
        NOTE2: hyperopt is minimizing, so use 100 - microF1, 100 - Acc, but EER!
        """
        self.model_cfg.update(params)
        print('=*' * 40)
        print('\n Trying Model parameters: ')
        pprint.pprint(self.model_cfg.toDict(), indent=2)

        # rebuild model from scratch
        self.model.rebuild(self.init_model_cfg)
        # TODO load pre-trained weights to optimize those Bayes space
        mfile = cfg.get_model_filename(path=path.join(self.config['path']['pretrain_set'], self.attrib_cls),
                                       train_mode=self.train_mode,
                                       fold=self.fold_id)
        self.model.load(mfile)
        self.model.chage_optimizer(self.model_cfg, change_out_unit=True)
        ######################

        # init batch generators
        train_gen = DL.batch_handler(batch_type=self.model_cfg['batch_type'],
                                     data_file=self.data.feat_file,
                                     fold_lst=self.data.meta_data.fold_list(self.fold_id, 'train'),
                                     config=self.model_cfg,
                                     meta_data=self.data.meta_data)

        cv_gen = DL.batch_handler(batch_type='sed_validation',
                                  data_file=self.data.feat_file,
                                  fold_lst=self.data.meta_data.fold_list(self.fold_id, 'validation'),
                                  config=self.model_cfg,
                                  meta_data=self.data.meta_data)

        def mfom_batch_wrap(xy_gen):
            for x, y in xy_gen.batch():
                yield [y, x], y

        wrap_gen = mfom_batch_wrap(train_gen) \
            if self.is_mfom_objective(self.model.model) \
            else train_gen.batch()
        self._init_callbacks(cv_gen)

        batch_sz = self.model_cfg['batch']
        samp_sz = train_gen.samples_number()
        print('Epoch size: %d observations' % samp_sz)

        # train model
        trial_result = {'params': self.model_cfg.toDict()}
        try:
            start = timer()
            self.model.model.fit_generator(wrap_gen,
                                           steps_per_epoch=samp_sz // batch_sz,
                                           nb_epoch=self.MAX_EPOCHS,
                                           verbose=self.verbose,
                                           workers=1,
                                           callbacks=self.callbacks)
            run_time = timer() - start
            # validate model
            mvals = ModelValidator.validate_model(model=self.model.model,
                                                  batch_gen=cv_gen,
                                                  metrics=self.model_cfg['metrics'])
            # loss of the hyperopt
            loss = mvals[self.config['callback']['monitor']]
            loss = loss if self.config['callback']['mode'] == 'min' else 100. - loss
            trial_result.update({
                'validation': mvals,
                'loss': loss,
                'time': run_time,
                'status': STATUS_OK
            })
            return trial_result
        except Exception as e:
            print('[error] Type: %s, %s' % (type(e), e.message))
            print('-' * 30)
            trial_result.update({
                'loss': 999999,
                'time': 0,
                'status': STATUS_FAIL
            })
            return trial_result
        finally:
            train_gen.stop()
            cv_gen.stop()

    def _init_callbacks(self, cv_gen):
        log_dir = cfg.get_log_dir(train_mode='hpsearch')
        print('LOG: %s' % log_dir)

        valtor = ModelValidator(batch_gen=cv_gen,
                                metrics=self.model_cfg['metrics'],
                                monitor=self.config['callback']['monitor'],
                                mode=self.config['callback']['mode'])
        lr_reductor = ReduceLROnPlateau(monitor=self.config['callback']['monitor'],
                                        patience=self.config['callback']['lr_patience'],
                                        verbose=self.verbose,
                                        factor=self.config['callback']['lr_factor'],
                                        min_lr=self.config['callback']['lr_min'])
        tboard = TensorBoard(log_dir=log_dir)
        self.callbacks = [valtor, lr_reductor, tboard]
