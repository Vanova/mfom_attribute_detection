import os.path as path
import pprint
import src.trainer.ogits_hyper as HP
import src.utils.config as cfg
import src.utils.dirs as dirs
from src.base.pipeline import BasePipeline
from src.data_loader.ogits import OGITSDataLoader
from src.model.sed_ogits import SEDOgitsModel
from src.trainer.ogits import OGITSModelTrainer


class OGITSApp(BasePipeline):
    def __init__(self, config, **kwargs):
        super(OGITSApp, self).__init__(config)
        self.pipe_mode = kwargs.get('pipe_mode', 'development')
        self.attrib_cls = kwargs.get('attrib_cls', 'place')
        self.verbose = kwargs.get('verbose', 2)
        self.overwrite = kwargs.get('overwrite', False)

        self.data_loader = OGITSDataLoader(self.config, pipe_mode=self.pipe_mode,
                                           attrib_cls=self.attrib_cls)

    def initialize_data(self):
        self.data_loader.initialize()

    def extract_feature(self):
        print('Feature params:')
        pprint.pprint(self.config['features'].toDict())
        feat_file = cfg.get_feature_filename(pipe_mode=self.pipe_mode,
                                             path=self.config['path']['features'])
        if not dirs.check_file(feat_file) or self.overwrite:
            self.data_loader.extract_features()
        else:
            print('[INFO] There is feature file: %s' % feat_file)

    def search_hyperparams(self):
        dirs.mkdir(self.config['path']['hyper_search'])
        # initialize default model
        nclass = len(self.data_loader.meta_data.label_names)
        model = self.inflate_model(self.config, nclass, train_mode='pretrain_set')

        hp_srch = HP.HyperSearch(model, self.data_loader, self.config,
                                 fold_id=1,
                                 train_mode='pretrain_set',
                                 attrib_cls=self.attrib_cls,
                                 verbose=self.verbose,
                                 overwrite=self.overwrite)
        hp_srch.train()

    def system_train(self):
        """
        Pre-training and fine-tuning logic
        """
        print('Model params:')
        pprint.pprint(self.config['model'].toDict())
        dirs.mkdirs(path.join(self.config['path']['finetune_set'], self.attrib_cls),
                    path.join(self.config['path']['pretrain_set'], self.attrib_cls))
        nclass = len(self.data_loader.meta_data.label_names)

        for fold_id in self.data_loader.meta_data.nfolds:
            if self.config['model']['do_pretrain']:
                print('/*========== Pre-training ==========*/')

                model = self.inflate_model(config=self.config, nclass=nclass)
                print('Pre-train with loss: %s' % model.model.loss)

                trainer = OGITSModelTrainer(model.model, self.data_loader, self.config,
                                            fold_id=fold_id,
                                            train_mode='pretrain_set',
                                            attrib_cls=self.attrib_cls,
                                            verbose=self.verbose,
                                            overwrite=self.overwrite)
                trainer.train()

            if self.config['model']['do_finetune']:
                print('/*========== Fine-tuning ==========*/')
                mfile = cfg.get_model_filename(path=path.join(self.config['path']['pretrain_set'], self.attrib_cls),
                                               train_mode='pretrain_set',
                                               fold=fold_id)
                pre_model = self.inflate_model(config=self.config, nclass=nclass)
                pre_model.load(mfile)
                pre_model.chage_optimizer(self.config['model']['finetune_set'], change_out_unit=True)
                print('Finetune with loss: %s' % pre_model.model.loss)

                finetuner = OGITSModelTrainer(pre_model.model, self.data_loader, self.config,
                                              fold_id=fold_id,
                                              train_mode='finetune_set',
                                              attrib_cls=self.attrib_cls,
                                              verbose=self.verbose,
                                              overwrite=self.overwrite)
                finetuner.train()

    def system_test(self):
        nclass = len(self.data_loader.meta_data.label_names)

        for fold_id in self.data_loader.meta_data.nfolds:
            for train_mode in ['pretrain_set', 'finetune_set']:
                print('/*========== Test %s model on FOLD %s ==========*/' % (train_mode.upper(), fold_id))

                model = self.inflate_model(self.config, nclass, train_mode=train_mode)
                mfile = cfg.get_model_filename(path=path.join(self.config['path'][train_mode], self.attrib_cls),
                                               train_mode=train_mode,
                                               fold=fold_id)
                model.load(mfile)
                print('Loss: %s' % model.model.loss)

                trainer = OGITSModelTrainer(model.model, self.data_loader, self.config,
                                            fold_id=fold_id,
                                            train_mode=train_mode,
                                            attrib_cls=self.attrib_cls,
                                            verbose=self.verbose,
                                            overwrite=self.overwrite)
                trainer.test()
                trainer.evaluate()

    @staticmethod
    def inflate_model(config, nclass, train_mode='pretrain_set'):
        batch_sz = config['model'][train_mode]['batch']
        bands = config['features']['bands']
        frame_wnd = config['model'][train_mode]['context_wnd']

        channels = 1
        if config['features']['include_delta']:
            channels += 1
        if config['features']['include_acceleration']:
            channels += 1
        feat_dim = (batch_sz, bands, frame_wnd, channels)
        return SEDOgitsModel(config['model'][train_mode],
                             input_shape=feat_dim,
                             nclass=nclass)
