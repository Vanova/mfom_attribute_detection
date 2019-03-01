import numpy as np
import os.path as path
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from keras.models import Model
from src.base.trainer import BaseTrainer
import src.utils.config as cfg
import src.utils.dirs as dirs
import src.utils.metrics as mtx
import src.data_loader.ogits as DL
import src.model.objectives as obj
import src.utils.io as io


class OGITSModelTrainer(BaseTrainer):
    """
    # Arguments
        model: Keras model, model for training
        data: BaseDataLoader, generate datasets (for train/test/validate)
        config: DotMap, settings for training 'model',
                according to 'training_mode' (i.e. pre-training/finetune)
        train_mode: String. 'pretrain_set' or 'finetune_set'.
        attrib_cls: String. Speech attribute 'manner', 'place' or 'fusion'
        verbose: Integer. 0, 1, or 2. Verbosity mode.
                    0 = silent, 1 = progress bar, 2 = one line per epoch.
        model: not trained or initially pre-trained model
    """

    def __init__(self, model, data, config, **kwargs):
        super(OGITSModelTrainer, self).__init__(model, data, config)

        self.train_mode = kwargs.get('train_mode', 'pretrain_set')
        self.fold_id = kwargs.get('fold_id', 1)
        self.attrib_cls = kwargs.get('attrib_cls', 'manner')
        self.verbose = kwargs.get('verbose', 1)
        self.overwrite = kwargs.get('overwrite', False)

        self.callbacks = []
        mpath = path.join(self.config['path'][self.train_mode], self.attrib_cls)
        self.model_file = cfg.get_model_filename(path=mpath,
                                                 train_mode=self.train_mode,
                                                 fold=self.fold_id)

    def _init_callbacks(self, cv_gen):
        self.callbacks.append(
            ModelValidator(batch_gen=cv_gen,
                           metrics=self.config['model'][self.train_mode]['metrics'],
                           monitor=self.config['callback']['monitor'],
                           mode=self.config['callback']['mode']))

        self.callbacks.append(
            ModelCheckpoint(
                monitor=self.config['callback']['monitor'],
                filepath=self.model_file,
                mode=self.config['callback']['mode'],
                save_best_only=self.config['callback']['chpt_save_best_only'],
                save_weights_only=self.config['callback']['chpt_save_weights_only'],
                verbose=self.verbose))

        self.callbacks.append(
            ReduceLROnPlateau(monitor=self.config['callback']['monitor'],
                              patience=self.config['callback']['lr_patience'],
                              verbose=self.verbose,
                              factor=self.config['callback']['lr_factor'],
                              min_lr=self.config['callback']['lr_min']))

        self.callbacks.append(
            EarlyStopping(monitor=self.config['callback']['monitor'],
                          patience=self.config['callback']['estop_patience'],
                          verbose=self.verbose,
                          mode=self.config['callback']['mode'],
                          min_delta=0.001))

        log_dir = cfg.get_log_dir(self.train_mode, self.fold_id)
        print('LOG: %s' % log_dir)
        self.callbacks.append(
            TensorBoard(log_dir=log_dir,
                        write_graph=self.config['callback']['tensorboard_write_graph']))

    @staticmethod
    def is_mfom_objective(model):
        return (model.loss in obj.MFOM_OBJECTIVES) or \
               (model.loss in obj.MFOM_OBJECTIVES.values())

    def train(self):
        if not dirs.check_file(self.model_file) or self.overwrite:
            # batch generators
            train_gen = DL.batch_handler(batch_type=self.config['model'][self.train_mode]['batch_type'],
                                         data_file=self.data.feat_file,
                                         fold_lst=self.data.meta_data.fold_list(self.fold_id, 'train'),
                                         config=self.config['model'][self.train_mode],
                                         meta_data=self.data.meta_data)

            cv_gen = DL.batch_handler(batch_type='sed_validation', #'sed_validation', 'validation'
                                      data_file=self.data.feat_file,
                                      fold_lst=self.data.meta_data.fold_list(self.fold_id, 'validation'),
                                      config=self.config['model'][self.train_mode],
                                      meta_data=self.data.meta_data)
            self._init_callbacks(cv_gen)

            smp_size = train_gen.samples_number()
            print('[INFO] Epoch size: %d observations' % smp_size)

            batch_sz = self.config['model'][self.train_mode]['batch']
            nepo = self.config['model'][self.train_mode]['n_epoch']

            def mfom_batch_wrap(xy_gen):
                for x, y in xy_gen.batch():
                    yield [y, x], y

            gen = mfom_batch_wrap(train_gen) \
                if self.is_mfom_objective(self.model) \
                else train_gen.batch()

            self.model.fit_generator(gen,
                                     steps_per_epoch=smp_size // batch_sz,
                                     nb_epoch=nepo,
                                     verbose=self.verbose,
                                     workers=1,
                                     callbacks=self.callbacks)
            train_gen.stop()
            cv_gen.stop()
        else:
            print('[INFO] There is %s model: %s' % (self.train_mode.upper(), self.model_file))

    def test(self):
        test_gen = DL.batch_handler(batch_type='sed_validation',
                                    data_file=self.data.feat_file,
                                    fold_lst=self.data.meta_data.fold_list(self.fold_id, 'test'),
                                    config=self.config['model'][self.train_mode],
                                    meta_data=self.data.meta_data)

        history = ModelValidator.validate_model(model=self.model,
                                                batch_gen=test_gen,
                                                metrics=self.config['model'][self.train_mode]['metrics'],
                                                attrib_cls=self.attrib_cls)
        print(history)
        test_gen.stop()

    def evaluate(self):
        test_gen = DL.batch_handler(batch_type='sed_evaluation',
                                    data_file=self.data.feat_file,
                                    fold_lst=self.data.meta_data.fold_list(self.fold_id, 'test'),
                                    config=self.config['model'][self.train_mode],
                                    meta_data=self.data.meta_data)

        cut_model = self.model
        if OGITSModelTrainer.is_mfom_objective(self.model):
            input = self.model.get_layer(name='input').output
            preact = self.model.get_layer(name='output').output
            cut_model = Model(input=input, output=preact)

        n_class = cut_model.output_shape[-1]
        cnt = 0
        scores = {}
        for fn, X_b in test_gen.batch():
            ps = cut_model.predict_on_batch(X_b)
            ps = ps.reshape((-1, n_class))
            fid = path.basename(fn)
            fid = path.splitext(fid)[0]
            scores[fid] = ps
            print('Processed: %d' % cnt)
            cnt += 1
        print('Evaluated files: %d' % cnt)
        io.save_ark('%s.ark' % self.attrib_cls, scores)


class ModelValidator(Callback):
    def __init__(self, batch_gen, metrics, monitor, mode):
        super(ModelValidator, self).__init__()
        self.batch_gen = batch_gen
        self.metrics = metrics
        self.monitor = monitor
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best_acc = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_acc = 0.
        else:
            raise AttributeError('[ERROR] ModelValidator mode %s is unknown')

    def on_train_begin(self, logs=None):
        super(ModelValidator, self).on_train_begin(logs)
        vs = ModelValidator.validate_model(self.model, self.batch_gen, self.metrics)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' BEFORE TRAINING: Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))
        print(logs)

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = -1

    def on_epoch_end(self, epoch, logs={}):
        super(ModelValidator, self).on_epoch_end(epoch, logs)
        vs = ModelValidator.validate_model(self.model, self.batch_gen, self.metrics)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' EPOCH %d. Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            epoch, vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))
        print(logs)

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        super(ModelValidator, self).on_train_end(logs)
        print('=' * 20 + ' Training report ' + '=' * 20)
        print('Best validation %s: epoch %s / %.4f\n' % (self.monitor.upper(), self.best_epoch, self.best_acc))

    @staticmethod
    def validate_model(model, batch_gen, metrics, attrib_cls=None):
        """
        # Arguments
            model: Keras model
            data: BaseDataLoader
            metrics: list of metrics
        # Output
            dictionary with values of metrics and loss
        """
        cut_model = model
        if OGITSModelTrainer.is_mfom_objective(model):
            input = model.get_layer(name='input').output
            preact = model.get_layer(name='output').output
            cut_model = Model(input=input, output=preact)

        n_class = cut_model.output_shape[-1]
        _, bnd, wnd, ch = batch_gen.batch_shape()
        y_true, y_pred = np.empty((0, wnd, n_class)), np.empty((0, wnd, n_class))

        loss, cnt = 0, 0
        for fn, X_b, Y_b in batch_gen.batch():
            ps = cut_model.predict_on_batch(X_b)
            y_pred = np.vstack([y_pred, ps])
            y_true = np.vstack([y_true, Y_b])
            # NOTE: it is fake loss, caz Y is fed
            if OGITSModelTrainer.is_mfom_objective(model):
                X_b = [Y_b, X_b]
            l = model.test_on_batch(X_b, Y_b)
            loss += l
            cnt += 1
        vals = {'val_loss': loss / cnt}
        y_true = y_true.reshape((-1, n_class))
        y_pred = y_pred.reshape((-1, n_class))

        if attrib_cls == 'fusion':
            fmanner_ids = [2, 4, 9, 11, 12, 14, 15]
            y_true_m = y_true[:, fmanner_ids]
            y_pred_m = y_pred[:, fmanner_ids]

            v_max, t_max = 0, 0
            for t in [0.5]: #np.linspace(0.2, 0.7, 50):
                p = mtx.step(y_pred_m, threshold=t)
                v = mtx.micro_f1(y_true_m, p)
                if v_max < v:
                    v_max = v
                    t_max = t
            print('F1 fusion-manner:', v_max, t_max)
            eer = np.mean(mtx.class_wise_eer(y_true_m, y_pred_m))
            print('EER fusion-manner:', eer)

            fplace_ids = [0, 1, 3, 5, 6, 7, 8, 10, 11, 13]
            y_true_p = y_true[:, fplace_ids]
            y_pred_p = y_pred[:, fplace_ids]

            v_max, t_max = 0, 0
            for t in [0.5]: # np.linspace(0.2, 0.7, 50):
                p = mtx.step(y_pred_p, threshold=t)
                v = mtx.micro_f1(y_true_p, p)
                if v_max < v:
                    v_max = v
                    t_max = t
            print('F1 fusion-place:', v_max, t_max)
            eer = np.mean(mtx.class_wise_eer(y_true_p, y_pred_p))
            print('EER fusion-place:', eer)

        for m in metrics:
            if m == 'micro_f1':
                v_max, t_max = 0, 0
                for t in [0.5]: #np.linspace(0.2, 0.7, 50):
                    p = mtx.step(y_pred, threshold=t)
                    v = mtx.micro_f1(y_true, p)
                    if v_max < v:
                        v_max = v
                        t_max = t
                vals[m] = v_max
                print('Optimal threshold: ', vals[m], t_max)
            elif m == 'pooled_eer':
                p = y_pred.flatten()
                y = y_true.flatten()
                vals[m] = mtx.eer(y, p)
            elif m == 'class_wise_eer':
                vals[m] = np.mean(mtx.class_wise_eer(y_true, y_pred))
            elif m == 'accuracy':
                p = np.argmax(y_pred, axis=-1)
                y = np.argmax(y_true, axis=-1)
                vals[m] = mtx.pooled_accuracy(y, p)
            else:
                raise KeyError('[ERROR] Such a metric is not implemented: %s...' % m)
        return vals
