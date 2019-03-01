import os
import fnmatch


def list_dir(root_dir):
    found_files = []
    for path, dirs, files in os.walk(root_dir):
        for f in files:
            found_files.append(os.path.join(path, f))
    return found_files


def mkdir(dir):
    try:
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
    except OSError as e:
        print("[ERR] Creating directory error: {0}".format(e))
        exit(-1)


def mkdirs(*dirs):
    """
    dirs: a list of directories to create if these are not found
    exit_code: 0:success -1:failed
    """
    for dir in dirs:
        mkdir(dir)


def check_file(file_name):
    return os.path.isfile(file_name)


def isempty(dir):
    """
    Check if the directory is empty.
    """
    return len(os.listdir(dir)) == 0


def search_files(search_dir, pattern='*.wav'):
    files_list = []
    for root, dirnames, filenames in os.walk(search_dir):
        for filename in fnmatch.filter(filenames, pat=pattern):
            files_list.append(os.path.join(root, filename))
    return files_list
