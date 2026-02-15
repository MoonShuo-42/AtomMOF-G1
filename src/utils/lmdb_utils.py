import lmdb
import pickle

def write_lmdb(lmdb_path):
    env = lmdb.open(
        str(lmdb_path), 
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=100 * 1024 * 1024 * 1024 # 100 GB
    )
    return env


def read_lmdb(lmdb_path, max_readers=32):
    env = lmdb.open(
        str(lmdb_path), 
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=max_readers,
    )
    return env

def get_data(txn, key):
    if isinstance(key, str):
        key = f"{key}".encode('ascii')
    value = txn.get(key)
    data = pickle.loads(value)
    return data

def get_all_keys(txn):
    keys = [bytes(key) for key, _ in txn.cursor()]
    return keys