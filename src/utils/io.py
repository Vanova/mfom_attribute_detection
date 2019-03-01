import h5py
import yaml
import csv
import numpy as np
import pandas as pd
import os.path as path


def load_csv(file_name, header=None, col_name=None, index_col=None, dtype=None, delim=','):
    """
    Read list of rows in csv, separated by delimiter
    # Output
        pandas DataFrame
    """
    res = pd.read_csv(file_name, header=header, names=col_name,
                      index_col=index_col, dtype=dtype, delimiter=delim)
    return res


def save_csv(file_name, list_data, delim=','):
    """Save list of rows to csv, separated by delim
    e.g. list_data=zip(l1, l2, l3)
    """
    with open(file_name, 'wt') as f:
        writer = csv.writer(f, delimiter=delim)
        for result_item in list_data:
            write_it = [it for it in result_item if it is not '']
            writer.writerow(write_it)


def load_txt(file_name):
    with open(file_name) as f:
        raw = f.readlines()
        lines = [s.strip() for s in raw]
    return lines


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        return yaml.load(f)


def load_dictionary(file_name, delim=' '):
    """
    Load text file as dictionary:
        1st column are keys, the rest columns are values
    """
    d = {}
    with open(file_name) as f:
        for line in f:
            sl = line.split(delim)
            clean_list = map(str.strip, sl)
            d[clean_list[0]] = clean_list[1:]
    return d


def load_mlf(file_name, lab_ext='lab'):
    """
    Load Master Label Files (MLF), e.g. applied in HTK toolkit or KALDI
        file_name: file path to the MLF
        lab_ext: extension of aligned audio files inside MLF file
    # Output
        dictionary: {file_id: list_of_events}
    """
    mlf_dict = {}
    with open(file_name) as f:
        header = f.next().strip()
        if header != '#!MLF!#':
            raise AttributeError('[ERROR] Header is not defined %s' % header)

        fid_now = None
        events = []
        for line in f:
            line = line.strip()
            if '.' + lab_ext in line:  # file alignment name
                fpath = line.replace('"', '')
                base = path.basename(fpath)
                fid_now = path.splitext(base)[0]
            elif line == '.':  # end of alignment
                mlf_dict[fid_now] = events
                events = []
                fid_now = None
            else:
                event = line.split()
                events.append(event)
    return mlf_dict


def save_mlf(file_name, mlf_dict, lab_ext='lab'):
    """
    Load Master Label Files (MLF), e.g. applied in HTK toolkit or KALDI
        file_name: file path to the MLF
        mlf_dict: dictionary. {file_id: list_of_events}
        lab_ext: extension of aligned files inside MLF file
    """
    header = '#!MLF!#\n'
    end = '.\n'
    with open(file_name, 'w') as f:
        f.write(header)

        for file_id, events in mlf_dict.items():
            fname = '"*/%s.%s"\n' % (file_id, lab_ext)
            f.write(fname)
            for ev in events:
                f.write(' '.join(ev) + '\n')
            f.write(end)


def save_ark(file_name, utt_feat):
    with open(file_name, 'w') as file:
        for utt, feats in utt_feat.items():
            head = "%s  [" % utt
            txt_feat = "\n  ".join(" ".join(str(i) for i in x) for x in feats)
            file.write("%s \n  %s ]\n" % (head, txt_feat))


class HDFWriter(object):
    """
    Save data features in single hdf5 file storage
        file_name: String. Hdf5 file storage name
    """

    def __init__(self, file_name):
        self.hdf = h5py.File(file_name, "w")

    def append(self, file_id, feat, tag=None):
        """
        file_id: unique identifier of the data feature file
        tag: hot-encoded 1D array, where '1' marks class on
        """
        if file_id in self.hdf.keys():
            print('[WARN] File already exists in the storage: %s' % file_id)
        else:
            # if file not exists then store it to hdf
            data = self.hdf.create_dataset(name=file_id, data=feat)
            if tag is not None:
                data.attrs['tag'] = tag

    def close(self):
        self.hdf.close()

    @staticmethod
    def load_data(file_name, keys=None):
        """
        Lazy load all datasets from hdf5 to the memory
        NOTE: not preferred to run for large dataset
        file_name: String. Path to hdf5 file storage
        keys: list. List of file ids
        """
        hdf = h5py.File(file_name, "r")
        if keys is None:
            files = list(hdf.keys())
            print('Files in dataset: %d' % len(files))
        else:
            files = keys
            print('Files by keys: %d' % len(files))

        X, Y = [], []
        for fn in hdf:
            X.append(np.array(hdf[fn]))
            Y.append(hdf[fn].attrs['tag'])
        hdf.close()
        return np.array(X), np.array(Y)


class ArkReader(object):
    START_MARK = '['
    END_MARK = ']'

    def __init__(self, data_file):
        self.data_file = data_file

    @staticmethod
    def grep(data_file, pattern='['):
        with open(data_file, 'r') as file:
            match = [line for line in file if pattern in line]
        return match

    @staticmethod
    def ark_len(data_file):
        match = ArkReader.grep(data_file, ArkReader.START_MARK)
        return len(match)

    def next_ark(self):
        """
        Read file with arks: every ark file one-by-one
        # Return
            file_id, features ndarray
        """
        cnt = 0
        with open(self.data_file, 'r') as file:
            utt = ''
            feats = []
            for line in file:
                # start of the features
                if ArkReader.START_MARK in line:
                    utt = line.strip().split()[0]
                    cnt += 1
                # end of the features
                elif ArkReader.END_MARK in line:
                    tmp = line.replace(ArkReader.END_MARK, '')
                    tmp = tmp.strip().split()
                    feats.append(tmp)
                    # return current ark file
                    yield utt, self._str2float(feats)
                    utt = ''
                    feats = []
                else:
                    # read features
                    feats.append(line.strip().split())

    def _str2float(self, arr):
        return np.array(arr).astype(np.float)


class ArkWriter(object):
    START_MARK = '['
    END_MARK = ']'

    def __init__(self, data_file):
        self.ark_file = open(data_file)

    def append(self, utt_feat):
        utt, feat = utt_feat
        head = '%s  [' % utt
        txt_feat = '\n  '.join(' '.join(str(i) for i in x) for x in feat)
        self.ark_file.write('%s \n  %s ]\n' % (head, txt_feat))

    def close(self):
        self.ark_file.close()
