import os
import time
import json
import hashlib
import io
from dotmap import DotMap

PARAM_DIR = 'params/'


class PipeMode(object):
    """
    DEV: for training, cross-validation and testing pipeline
    SUBMIT: for evaluation pipeline
    """
    DEV = 'development'
    SUBMIT = 'submission'


def today_date():
    return time.strftime('%d_%b_%Y')


def now_time():
    return time.strftime('%H:%M:%S')


def verbosity(args):
    v = 2
    if args['--quiet']:
        v = 0
    elif args['--verbose']:
        v = 1
    return v


def get_log_dir(train_mode, fold=None):
    """
    train_mode: String. 'pretrain_set', 'finetune_set', 'hpsearch'
    fold: Integer. dataset fold
    """
    if fold:
        return 'logs/%s/fold%d/%s/%s' % (today_date(), fold, train_mode, now_time())
    else:
        return 'logs/%s/%s/%s' % (today_date(), train_mode, now_time())


def get_feature_filename(pipe_mode, path, fname='full', fold=None, extension='hdf5'):
    """
    pipe_mode: String. pipe mode: 'development', 'submission'
    """
    fn = str(pipe_mode) + '_feat_' + fname
    if fold:
        return os.path.join(path, '%s_fold%d.%s' % (fn, fold, extension))
    else:
        return os.path.join(path, '%s.%s' % (fn, extension))


def get_model_filename(path, train_mode, fold=None, extension='hdf5'):
    """
    path: root path
    train_mode: String. 'pretrain_set', 'finetune_set', 'hpsearch'
    fold: Integer. number of data fold
    """
    if fold:
        return os.path.join(path, '%s_model_fold%d.%s' % (train_mode, fold, extension))
    else:
        return os.path.join(path, '%s_model.%s' % (train_mode, extension))


def get_train_result_filename(fold, path, extension='txt'):
    """File name with network output scores on training set"""
    return os.path.join(path, 'train_scores_fold%d.%s' % (fold, extension))


def get_test_result_filename(fold, path, set_type='test', extension='txt'):
    """File name with network output scores on test set"""
    return os.path.join(path, '%s_scores_fold%d.%s' % (set_type, fold, extension))


def get_train_Y_filename(fold, path, extension='txt'):
    """Ground truth of training set"""
    return os.path.join(path, 'train_Y_fold%d.%s' % (fold, extension))


def get_test_Y_filename(fold, path, extension='txt'):
    """Ground truth of test set"""
    return os.path.join(path, 'test_Y_fold%d.%s' % (fold, extension))


def get_evaluation_filename(path, extension='txt'):
    """Ground truth of test set"""
    return os.path.join(path, 'evaluation_result.%s' % extension)


def process_parameters(p_file):
    """
    Create path structure for experiment:
    system/<experiment_name>/features/<feat_params_hash>/
    system/<experiment_name>/hyper_search/<feat_params_hash>/<hyper_search_hash>/
    system/<experiment_name>/model/<feat_params_hash>/<pretrain_params_hash>/<finetune_params_hash>/
    system/<experiment_name>/results_train/<feat_params_hash>/<pretrain_params_hash>/<finetune_params_hash>/
    system/<experiment_name>/results_evaluate/<feat_params_hash>/<pretrain_params_hash>/<finetune_params_hash>/
    system/<experiment_name>/results_ensemble/<feat_params_hash>/<ensemble_hash>/

    # Attributes
        p_file: yaml. parameters file
    # Return
        process and calculate hash on parameters
    """
    p = io.load_yaml(p_file)
    p = DotMap(p)
    # ===
    # Feature parameters,
    # NOTE: if parameters are changed, then paths with new hash values are created
    # ===
    feat_type = p['features']['type']
    # choose only particular feature type
    p['features'] = p['features'][feat_type]
    p['features']['type'] = feat_type
    # feature hash
    p['features']['hash'] = get_parameter_hash(p['features'])
    # ===
    # Model parameters
    # ===
    model_type = p['model']['type']
    # choose settings only of the particular model type
    p['model'] = p['model'][model_type]
    p['model']['type'] = model_type
    # pre-training model hash
    p['model']['pretrain_set']['hash'] = get_parameter_hash(p['model']['pretrain_set'])
    # tuning model hash
    p['model']['finetune_set']['hash'] = get_parameter_hash(p['model']['finetune_set'])
    # ===
    # Ensemble parameters
    # ===
    p['ensemble']['hash'] = get_parameter_hash(p['ensemble'])

    # ===
    # Add hashes to the paths
    # ===
    p['path']['meta'] = os.path.join(p['path']['base'],
                                     p['experiment']['name'],
                                     p['path']['meta'])

    p['path']['features'] = os.path.join(p['path']['base'],
                                         p['experiment']['name'],
                                         p['path']['features'],
                                         p['features']['hash'])

    p['path']['hyper_search'] = os.path.join(p['path']['base'],
                                             p['experiment']['name'],
                                             p['path']['hyper_search'],
                                             p['features']['hash'])

    p['path']['pretrain_set'] = os.path.join(p['path']['base'],
                                             p['experiment']['name'],
                                             p['path']['models'],
                                             p['features']['hash'],
                                             p['model']['pretrain_set']['hash'])

    # save tuned model into the pretrained model directory
    p['path']['finetune_set'] = os.path.join(p['path']['pretrain_set'],
                                             p['model']['finetune_set']['hash'])

    p['path']['train_result'] = os.path.join(p['path']['base'],
                                             p['experiment']['name'],
                                             p['path']['train_result'],
                                             p['features']['hash'],
                                             p['model']['pretrain_set']['hash'],
                                             p['model']['finetune_set']['hash'])

    p['path']['eval_result'] = os.path.join(p['path']['base'],
                                            p['experiment']['name'],
                                            p['path']['eval_result'],
                                            p['features']['hash'],
                                            p['model']['pretrain_set']['hash'],
                                            p['model']['finetune_set']['hash'])

    p['path']['submission'] = os.path.join(p['path']['base'],
                                           p['experiment']['name'],
                                           p['path']['submission'],
                                           p['features']['hash'],
                                           p['model']['pretrain_set']['hash'],
                                           p['model']['finetune_set']['hash'])
    return p


def get_parameter_hash(params):
    md5 = hashlib.md5()
    md5.update(str(json.dumps(params, sort_keys=True)))
    return md5.hexdigest()
