import time
import csv
from collections import Counter, Sequence

import numpy as np

_order_types = ['frequency', 'alphabet']


def build_vocabs(tokens, order_by=None):
    if order_by is None:
        order_by = 'frequency'
    if order_by == 'frequency':
        counter = Counter(tokens)
        vocab, counts = zip(*sorted(counter.items(), key=lambda e: e[1], reverse=True))
    elif order_by == 'alphabet':
        vocab = set(tokens)
        vocab = sorted(vocab)
    else:
        raise ValueError('order_by must be one of {}'.format(_order_types))
    return list(vocab)


def map_nested_list(nested_list, f, max_depth=None):
    if max_depth == 0:
        return f(nested_list)
    if isinstance(nested_list, Sequence):
        new_depth = max_depth if max_depth is None else (max_depth-1)
        return [map_nested_list(e, f, max_depth=new_depth) for e in nested_list]
    return f(nested_list)


def read_csv(csv_path, col_processors=None, header=False, verbose=True, **csv_args):
    start = time.time()

    res = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        if header:
            f.readline()
        reader = csv.reader(f, **csv_args)
        for row in reader:
            if col_processors is None:
                res.append(row)
            else:
                res.append([p(row[i]) for i, p in enumerate(col_processors) if p is not None])
         
    if verbose:
        print('reading {} rows in {:.2f}s'.format(len(res), time.time() - start, ))
    return res


def train_test_split(arr, train_size=0.8):
    rand_idx = np.arange(len(arr))
    np.random.shuffle(rand_idx)
    split = int(train_size * len(rand_idx))
    train_idx, test_idx = rand_idx[:split], rand_idx[split:]
    if isinstance(arr, np.ndarray):
        return arr[train_idx], arr[test_idx]
    return [arr[i] for i in train_idx], [arr[i] for i in test_idx]