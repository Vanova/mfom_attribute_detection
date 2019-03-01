"""
    Speech Attributes Detection Project
    =====================================
    Project for speech articulatory attributes detection,
        such as manner and place of articulation.
    These features are used for spoken language recognition, accent detection.
    This project is based on OGI-TS dataset.
    --------------------------------------------------------------------
        Affiliation: University of Eastern Finland / The SIPU Lab, Finland
                     A*Star / I2R, Singapore
        Author:  Ivan Kukanov

    Usage:
        run_ogits.py [-p FILE] [-m (dev|sub)] [-a CLASS] [--overwrite] [--quiet | --verbose]
        run_ogits.py -m dev -p file.yaml --verbose
        run_ogits.py -m dev -p file.yaml -a manner --verbose
        run_ogits.py -m sub -p file.yaml
        run_ogits.py (-s | --show_params)
        run_ogits.py (-h | --help)
        run_ogits.py (-v | --version)

    Options:
        -h --help               Show this screen.
        -v --version            Show version.
        -s --show_params        Show available configuration files  [default: False].
        -m --mode               System mode: development (dev) / submission or evaluation (sub) [default: dev].
        -p FILE, --params FILE  Model configuration file            [default: params/ogits.yaml].
        -a CLASS,--attribute CLASS  Speech attribute class: place, manner, fusion
        --overwrite             Force to overwrite model data       [default: False].
        --quite                 Output less log information         [default: False].
        --verbose               Output detailed log information     [default: False].
"""
from __future__ import print_function, absolute_import
import os
import sys
import numpy as np
from docopt import docopt
import src.utils.config as cfg
import src.utils.dirs as dirs
from src.pipeline.ogits import OGITSApp

np.random.seed(777)
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
__version_info__ = ('0', '0', '1')
__version__ = '.'.join(__version_info__)


def main(args):
    if args['--show_params']:
        lst = dirs.list_dir(cfg.PARAM_DIR)
        print('Parameter files found:', lst)
        return

    params = cfg.process_parameters(args['--params'])
    v = cfg.verbosity(args)

    if args['--mode'] and args['dev']:
        attribute = args['--attribute'] if args['--attribute'] \
            else params['experiment']['attribute_class']

        app = OGITSApp(params, pipe_mode=cfg.PipeMode.DEV, attrib_cls=attribute,
                       verbose=v, overwrite=args['--overwrite'])

        if params['pipeline']['init_dataset']:
            app.initialize_data()

        if params['pipeline']['extract_features']:
            app.extract_feature()

        if params['pipeline']['search_hyperparams']:
            app.search_hyperparams()

        if params['pipeline']['train_system']:
            app.system_train()

        if params['pipeline']['test_system']:
            app.system_test()

    elif args['--mode'] and args['sub']:
        print('Submission could be run here...')


if __name__ == '__main__':
    arguments = docopt(__doc__, version=__version__)
    main(arguments)
