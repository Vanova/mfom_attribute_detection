import os.path as path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import bisect
import src.utils.io as io
import src.utils.dirs as dirs
import src.utils.config as cfg
from src.base.data_loader import BaseDataLoader, BaseMeta, BaseGenerator
import src.features.speech as F


class OGITSDataLoader(BaseDataLoader):
    """
    Dataset handler facade.
    Arguments:
        config: dict. Parameters of data loader.
        pipe_mode: String. 'development' or 'submission'
    """

    def __init__(self, config, pipe_mode, **kwargs):
        super(OGITSDataLoader, self).__init__(config)
        self.pipe_mode = pipe_mode
        self.attrib_cls = kwargs.get('attrib_cls', 'place')

        self.feat_file = cfg.get_feature_filename(pipe_mode=pipe_mode,
                                                  path=self.config['path']['features'])

        meta_paths = [path.join(self.config['path']['meta'], at) for at in OGITSDev.ATTRIBUTES]
        dirs.mkdirs(self.config['path']['features'], *meta_paths)
        self._meta = None

    @property
    def meta_data(self):
        if not self._meta:
            raise ValueError('[error] Not initialized meta data.')
        return self._meta

    def initialize(self):
        print('[INFO] Preprocess meta data...')
        attrib_dir = path.join(self.config['path']['meta'], self.attrib_cls)
        if dirs.isempty(attrib_dir):
            self._phoneme_to_attribute()
            self._split_mlf_list()
            self._transform_mlf_to_dcase()

        if self.pipe_mode == cfg.PipeMode.DEV:
            self._meta = OGITSDev(data_dir=self.config['experiment']['development_dataset'],
                                  lists_dir=self.config['experiment']['lists_dir'],
                                  meta_dir=self.config['path']['meta'],
                                  feat_conf=self.config['features'],
                                  attrib_cls=self.attrib_cls)
        elif self.pipe_mode == cfg.PipeMode.SUBMIT:
            print('[INFO] There could be your evaluation data...')

    def extract_features(self):
        print('[INFO] Extract features from all audio files')

        all_ali = self.meta_data.file_list()
        fnames = set(all_ali[0])

        # prepare extractor
        feat_type = self.config['features']['type']
        extractor = F.prepare_extractor(feats=feat_type, params=self.config['features'])
        writer = io.HDFWriter(file_name=self.feat_file)
        cnt = 0
        for fn in fnames:
            file_path = path.join(self.meta_data.data_dir, fn)
            x, fs = F.load_sound_file(file_path)
            feat = extractor.extract(x, fs)
            # dump features
            writer.append(file_id=path.basename(fn), feat=feat)
            cnt += 1
            print("%d. processed: %s" % (cnt, fn))
        writer.close()
        print("Files processed: %d" % len(fnames))

    def train_data(self):
        raise NotImplementedError('[INFO] This is lazy method for small datasets.'
                                  'Use MetaData and BaseGenerator.')

    def test_data(self):
        raise NotImplementedError('[INFO] This is lazy method for small datasets.'
                                  'Use MetaData and BaseGenerator.')

    def eval_data(self):
        raise NotImplementedError('[INFO] This is lazy method for small datasets.'
                                  'Use MetaData and BaseGenerator.')

    def _phoneme_to_attribute(self):
        print('[INFO] Mapping phonemes to attributes')
        lists_dir = self.config['experiment']['lists_dir']

        for att in OGITSDev.ATTRIBUTES:
            att_align = {}
            for lang in OGITSDev.LANGUAGES:
                mfile = path.join(lists_dir, 'mapping', att, '%s_ph_at.txt' % lang)
                ph_att = io.load_dictionary(mfile)

                afile = path.join(lists_dir, 'phonemes', '%s.mlf' % lang)
                ph_align = io.load_mlf(afile)

                for file_id, events in ph_align.items():
                    for ev in events:
                        labels = ph_att[ev[-1]]
                        ev[-1] = ';'.join(labels)

                print('%s: language %s, # of alignments %d' % (att, lang, len(ph_align)))
                att_align.update(ph_align)

            ali_f = path.join(self.config['path']['meta'], att, 'alignment.mlf')
            io.save_mlf(ali_f, att_align)

    def _split_mlf_list(self):
        print('[INFO] Split alignment lists: tran/test/validation')
        lists_dir = self.config['experiment']['lists_dir']
        meta_dir = self.config['path']['meta']

        for data_type in OGITSDev.DATA_TYPES:
            wav_list = []
            with open(path.join(lists_dir, 'lists', data_type)) as f:
                lines = map(str.strip, f.readlines())
                wav_list.extend(lines)
                wav_list.sort()
                wav_list = map(lambda x: x.replace('.wav', ''), wav_list)

            for attrib in OGITSDev.ATTRIBUTES:
                attrib_f = path.join(meta_dir, attrib, 'alignment.mlf')
                alignment = io.load_mlf(attrib_f)
                sub = dict((f, alignment[f]) for f in wav_list)
                sub_f = path.join(meta_dir, attrib, '%s.mlf' % data_type)
                io.save_mlf(sub_f, sub)

    def _transform_mlf_to_dcase(self):
        meta_dir = self.config['path']['meta']
        for attrib in OGITSDev.ATTRIBUTES:
            for data_type in OGITSDev.DATA_TYPES:
                ali_f = path.join(meta_dir, attrib, '%s.mlf' % data_type)
                alignment = io.load_mlf(ali_f)

                res = self._mlf_dcase(alignment)
                res.reset_index(drop=True, inplace=True)
                csv_f = path.join(meta_dir, attrib, 'fold1_%s.csv' % data_type)
                res.to_csv(csv_f, index=False, header=False, sep='\t')

    def _mlf_dcase(self, mlf_ali):
        search_dir = self.config['experiment']['development_dataset']
        wav_paths = dirs.search_files(search_dir)

        res = pd.DataFrame()
        for fid, events in mlf_ali.items():
            wav_f = filter(lambda x: fid in x, wav_paths)
            if len(wav_f) > 1:
                raise Exception('Found files: %s' % wav_f)
            elif len(wav_f) == 0:
                raise Exception('File is not found!!!')

            wav_f = wav_f[0].replace(search_dir + '/', '')
            starts = np.array(events)[:, 0].astype(int) / 10. ** 7
            ends = np.array(events)[:, 1].astype(int) / 10. ** 7
            labs = np.array(events)[:, 2]
            df_ali = pd.DataFrame(zip([wav_f] * len(starts), starts, ends, labs),
                                  columns=['file', 'start', 'end', 'class_label'])
            res = res.append(df_ali)
        return res


def batch_handler(batch_type, data_file, config, fold_lst=None, meta_data=None):
    """
    batch_type:
        'sed_sequence' - slice chunks sequentially from the taken file
        'sed_random_crop' -
        'sed_validation' -
        'sed_evaluation' -
    config: current model configuration
    """
    batch_sz = config['batch']
    wnd = config['context_wnd']
    print('Batch type: %s' % batch_type)

    if batch_type == 'sed_sequence':
        return SEDSequenceGenerator(data_file=data_file,
                                    batch_sz=batch_sz,
                                    window=wnd,
                                    fold_list=fold_lst,
                                    meta_data=meta_data,
                                    config=config,
                                    shuffle=False)
    elif batch_type == 'sed_random_crop':
        return SEDRandomGenerator(data_file=data_file,
                                  batch_sz=batch_sz,
                                  window=wnd,
                                  fold_list=fold_lst,
                                  meta_data=meta_data,
                                  config=config,
                                  shuffle=False)
    elif batch_type == 'sed_validation':
        return SEDValidationGenerator(data_file=data_file,
                                      batch_sz=batch_sz,
                                      window=wnd,
                                      fold_list=fold_lst,
                                      meta_data=meta_data,
                                      config=config,
                                      shuffle=False)
    elif batch_type == 'sed_evaluation':
        return SEDEvaluationGenerator(data_file=data_file,
                                      batch_sz=batch_sz,
                                      window=wnd,
                                      fold_list=fold_lst,
                                      meta_data=meta_data,
                                      config=config,
                                      shuffle=False)
    else:
        raise ValueError('Unknown batch type [' + batch_type + ']')


class SequentialGenerator(BaseGenerator):
    """Generate sequence of observations."""

    def __init__(self, data_file, batch_sz=1, window=0, fold_list=None, meta_data=None, **kwargs):
        super(SequentialGenerator, self).__init__(data_file, batch_sz, window, fold_list)
        self.meta_data = meta_data  # meta is needed if we need to calculate data stats
        self.config = kwargs['config']
        self.hop_step = self.window  # 1

    def batch(self):
        """
        Slice HDF5 datasets and return batch: [smp x bands x frames [x channel]]
        """
        while 1:
            count_smp = 0
            X, Y = [], []
            # slide with window frame-by-frame across all the file
            fnames = list(set(self.fold_list[0]))
            np.random.shuffle(fnames)

            for fn in fnames:
                fid = path.basename(fn)
                feat = np.array(self.hdf[fid], dtype=np.float32)
                dim, N, ch = feat.shape

                # choose events by current file name
                events = filter(lambda x: x[0] == fn, zip(*self.fold_list))
                _, starts, ends, labs = zip(*events)

                for frame in xrange(0, N, self.hop_step):
                    # Get start and end of the window, keep frame at the middle (approximately)
                    start_frame = int(frame - np.floor(self.window / 2.))
                    end_frame = int(frame + np.ceil(self.window / 2.))

                    f_ids = np.array(range(start_frame, end_frame))
                    # pad with first frame
                    f_ids[f_ids < 0] = 0
                    # pad with last frame
                    f_ids[f_ids > N - 1] = N - 1
                    now_sample = feat[:, f_ids, :]

                    lab_id = bisect.bisect_left(ends, min(frame, ends[-1]))
                    now_lab = labs[lab_id]

                    if count_smp < self.batch_sz:
                        count_smp += 1
                    else:
                        X = np.array(X)
                        Y = np.array(Y)
                        yield X, Y
                        count_smp = 0
                        X, Y = [], []
                    X.append(now_sample)
                    Y.append(now_lab)

    def batch_shape(self):
        """
        NOTE: input dataset always is in Tensorflow order [bands, frames, channels]
        # Return:
            Tensorflow:
                3D data [batch_sz; band; frame_wnd; channel]
        """
        fn = path.basename(self.fold_list[0][0])
        sh = np.array(self.hdf[fn]).shape
        if len(sh) == 3:
            bands, _, channels = sh
            assert channels >= 1
            return self.batch_sz, bands, self.window, channels

    def samples_number(self):
        """
        return: number of total observations, i.e. N_files * file_len/wnd
        """
        total_smp = 0
        for fn in set(self.fold_list[0]):
            dim, N, ch = self.hdf[path.basename(fn)].shape
            total_smp += N // self.hop_step
        return total_smp


class SEDSequenceGenerator(BaseGenerator):
    """Generate sequence of observations."""

    def __init__(self, data_file, batch_sz=1, window=0, fold_list=None, meta_data=None, **kwargs):
        super(SEDSequenceGenerator, self).__init__(data_file, batch_sz, window, fold_list)
        self.meta_data = meta_data
        self.config = kwargs['config']
        self.hop_step = self.window // 2  # 1

    def batch(self):
        """
        Slice HDF5 datasets and return batch: [smp x bands x frames [x channel]]
        """
        while 1:
            # slide with window frame-by-frame across all the file
            fnames = list(set(self.fold_list[0]))
            np.random.shuffle(fnames)

            n_class = len(self.meta_data.label_names)
            count_smp = 0
            X, Y = [], []
            for fn in fnames:
                if count_smp < self.batch_sz:
                    count_smp += 1
                else:
                    yield np.array(X), np.array(Y)
                    count_smp = 0
                    X, Y = [], []
                fid = path.basename(fn)
                feat = np.array(self.hdf[fid], dtype=np.float32)
                dim, n_frames, ch = feat.shape
                # sample feature window
                last = (n_frames - self.window) // self.hop_step * self.hop_step
                start = np.random.randint(0, last)
                X.append(feat[:, start:start + self.window, :])
                # prepare labels
                # choose events by current file name
                events = filter(lambda x: x[0] == fn, zip(*self.fold_list))
                _, _, ends, labs = zip(*events)
                lseq = np.empty((0, n_class))
                for id in xrange(start, start + self.window):
                    lab_id = bisect.bisect_left(ends, min(id, ends[-1]))
                    lseq = np.vstack([lseq, labs[lab_id]])
                Y.append(lseq)

    def samples_number(self):
        """
        return: number of total observations, i.e. N_files * file_len/wnd
        """
        total_smp = 0
        for fn in set(self.fold_list[0]):
            dim, N, ch = self.hdf[path.basename(fn)].shape
            total_smp += N // self.hop_step
        return total_smp


class SEDRandomGenerator(BaseGenerator):
    """Generate sequence of observations."""

    def __init__(self, data_file, batch_sz=1, window=0, fold_list=None, meta_data=None, **kwargs):
        super(SEDRandomGenerator, self).__init__(data_file, batch_sz, window, fold_list)
        self.meta_data = meta_data  # meta is needed if we need to calculate data stats
        self.config = kwargs['config']
        self.hop_step = self.window // 2  # 1

    def batch(self):
        """
        Slice HDF5 datasets and return batch: [smp x bands x frames [x channel]]
        """
        while 1:
            # slide with window frame-by-frame across all the file
            fnames = list(set(self.fold_list[0]))
            np.random.shuffle(fnames)

            n_class = len(self.meta_data.label_names)
            count_smp = 0
            X, Y = [], []
            for fn in fnames:
                fid = path.basename(fn)
                feat = np.array(self.hdf[fid], dtype=np.float32)
                dim, n_frames, ch = feat.shape

                # choose events by current file name
                events = filter(lambda x: x[0] == fn, zip(*self.fold_list))
                _, starts, ends, labs = zip(*events)

                last = (n_frames - self.window) // self.hop_step * self.hop_step

                for start in xrange(0, last, self.hop_step):
                    if count_smp < self.batch_sz:
                        count_smp += 1
                    else:
                        yield np.array(X), np.array(Y)
                        count_smp = 1
                        X, Y = [], []
                    X.append(feat[:, start:start + self.window, :])
                    # prepare label
                    lseq = np.empty((0, n_class))
                    for id in xrange(start, start + self.window):
                        lab_id = bisect.bisect_left(ends, min(id, ends[-1]))
                        lseq = np.vstack([lseq, labs[lab_id]])
                    Y.append(lseq)

    def samples_number(self):
        """
        return: number of total observations, i.e. N_files * file_len/wnd
        """
        total_smp = 0
        for fn in set(self.fold_list[0]):
            dim, N, ch = self.hdf[path.basename(fn)].shape
            total_smp += N // self.hop_step
        return total_smp


class SEDValidationGenerator(SequentialGenerator):
    def batch(self):
        # slide with window frame-by-frame across all the file
        self.hop_step = self.window
        n_class = len(self.meta_data.label_names)
        fnames = list(set(self.fold_list[0]))
        count_smp = 0
        for fn in fnames:
            X, Y = [], []
            fid = path.basename(fn)
            feat = np.array(self.hdf[fid], dtype=np.float32)
            dim, n_frames, ch = feat.shape

            # events of the file fn
            events = filter(lambda x: x[0] == fn, zip(*self.fold_list))
            _, starts, ends, labs = zip(*events)
            last_frm = n_frames // self.hop_step * self.hop_step

            for start in xrange(0, last_frm, self.hop_step):
                # if count_smp < self.batch_sz:
                #     count_smp += 1
                # else:
                #     yield np.array(X), np.array(Y)
                #     count_smp = 1
                #     X, Y = [], []
                X.append(feat[:, start:start + self.window, :])
                # prepare label
                lseq = np.empty((0, n_class))
                for id in xrange(start, start + self.window):
                    lab_id = bisect.bisect_left(ends, min(id, ends[-1]))
                    lseq = np.vstack([lseq, labs[lab_id]])
                Y.append(lseq)
            yield fn, np.array(X), np.array(Y)


class SEDEvaluationGenerator(BaseGenerator):
    def __init__(self, data_file, batch_sz=1, window=0, fold_list=None, meta_data=None, **kwargs):
        super(SEDEvaluationGenerator, self).__init__(data_file, batch_sz, window, fold_list)
        self.meta_data = meta_data
        self.config = kwargs['config']
        self.hop_step = self.window
        print('NOTE: it is generating batch from single file')

    def batch(self):
        fnames = list(set(self.fold_list[0]))
        for fn in fnames:
            fid = path.basename(fn)
            feat = np.array(self.hdf[fid], dtype=np.float32)
            dim, n_frames, ch = feat.shape

            # events of the file fn
            events = filter(lambda x: x[0] == fn, zip(*self.fold_list))
            _, starts, ends, labs = zip(*events)
            last_frm = n_frames // self.hop_step * self.hop_step

            X = []
            for start in xrange(0, last_frm, self.hop_step):
                X.append(feat[:, start:start + self.window, :])
            yield fn, np.array(X)


class OGITSDev(BaseMeta):
    """
    OGI-TS dataset meta information: labels, alignments,...
    Attribute alignments are in in DCASE format.
    DCASE format:
        ['file_id' 'start' 'end' 'class_label']
        e.g. [file.wav 12.152502 13.195551 labial]

    # Arguments
        data_dir: path to dataset
        lists_dir: path to meta data lists: attribute classes,
            mapping phonemes to attributes, alignments
        meta_dir: output path of processed meta data
        feat_conf: dict. Feature configuration
    """
    ATTRIBUTES = ['manner', 'place', 'fusion']
    LANGUAGES = ['english',
                 'german',
                 'hindi',
                 'japanese',
                 'mandarin',
                 'spanish']
    DATA_TYPES = ['train', 'validation', 'test']

    def __init__(self, data_dir, lists_dir, feat_conf, meta_dir, attrib_cls='manner'):
        self.data_dir = data_dir
        self.lists_dir = lists_dir
        self.feat_conf = feat_conf
        self.meta_dir = path.join(meta_dir, attrib_cls)
        self.attrib_cls = attrib_cls

        if self.attrib_cls not in self.ATTRIBUTES:
            raise AttributeError('[ERROR] There is no such an attribute type: %s' % attrib_cls)

        self.col = ['file', 'start', 'end', 'class_label']
        self._meta_file_template = 'fold{num}_{dtype}.csv'
        self.folds_num = 1  # we have one fold of data

        self.lencoder = self._labels_encoder()

    @property
    def label_names(self):
        return list(self.lencoder.classes_)

    @property
    def nfolds(self):
        """Return list of folds indices"""
        return range(1, self.folds_num + 1)

    def labels_str_encode(self, lab_dig):
        """
        Transform hot-vector to 'class_id' format
        lab_dig: list.
        """
        return list(self.lencoder.inverse_transform(lab_dig))

    def labels_dig_encode(self, lab_str):
        """
        Transform 'class_id' to hot-vector format
        lab_str: list.
        """
        return self.lencoder.transform(lab_str)

    def fold_list(self, fold, data_type):
        """
        fold: Integer. Number of fold to return
        data_type: String. 'train', 'test', 'validation', this is the development lists
        return: list. File names and event alignments
        """
        if not (data_type in self.DATA_TYPES):
            raise AttributeError('[ERROR] No dataset type: %s' % data_type)
        flist = self._load_fold_list(fold, data_type, self.meta_dir)
        return self._format_list(flist)  # file_name, start_frame, end_frame, hot_vecs

    def file_list(self):
        """
        Joint file list of the development dataset (training + cv lists),
        return: lists, file names and labels digital binary encoding
        """
        # merge train & test lists in general data
        mrg = pd.DataFrame()
        for dt in self.DATA_TYPES:
            fl = self._load_fold_list(fold=1, data_type=dt, dir=self.meta_dir)
            mrg = mrg.append(fl)
        mrg.reset_index(drop=True, inplace=True)
        return self._format_list(mrg)  # file_name, start_frame, end_frame, hot_vecs

    def _labels_encoder(self):
        """
        prepare labels encoder from string to digits
        """
        meta_file = path.join(self.meta_dir,
                              self._meta_file_template.format(num=1, dtype=self.DATA_TYPES[0]))
        pd_meta = io.load_csv(meta_file, col_name=self.col, delim='\t')
        # labels transform
        le = MultiLabelBinarizer()
        le.fit(pd_meta[self.col[-1]].str.split(';'))
        return le

    def _load_fold_list(self, fold, data_type, dir):
        meta_file = path.join(dir, self._meta_file_template.format(num=fold, dtype=data_type))
        if path.isfile(meta_file):
            pd_meta = io.load_csv(meta_file, col_name=self.col, delim='\t')
            return pd_meta
        else:
            raise Exception('[error] file %s do not exist!' % meta_file)

    def _format_list(self, df):
        starts = df[self.col[1]].values
        ends = df[self.col[2]].values
        resolution = self.feat_conf['hop_length_seconds']

        fnames = list(df[self.col[0]])
        # onset/offset time stamps to frame ids
        start_ids = np.floor(starts * 1.0 / resolution).astype(int).tolist()
        end_ids = np.floor(ends * 1.0 / resolution).astype(int).tolist()
        # transform 'class_label' to hot-vectors
        labs = df[self.col[3]].str.split(';')
        hot_vecs = self.labels_dig_encode(labs)
        return fnames, start_ids, end_ids, hot_vecs
