import math
import numpy as np
from operator import itemgetter
import os
import pandas as pd
import re

UNK_TOKEN = '<UNK>'


def calc_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def create_vocab(values):
    vocab = {UNK_TOKEN: 0}
    i = 1
    for val in set(values):
        if val is not None and val != UNK_TOKEN:
            vocab[str(val)] = i
            i += 1

    return vocab


RARE_TITLES = {'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
               'Rev', 'Sir', 'Jonkheer', 'Dona'}


def get_title(name):
    match = re.search(r',\s+([A-Za-z]+)\.', name)
    if match:
        title = match.group(1)
        if title in RARE_TITLES:
            return 'Rare'
        elif title == 'Mlle' or title == 'Miss':
            return 'Ms'
        elif title == 'Mme':
            return 'Mrs'
        else:
            return title

    return UNK_TOKEN


def get_vocab(col, values):
    filename = str(col) + '_vocab.txt'
    if os.path.exists(filename):
        return load_vocab(filename)

    vocab = create_vocab(values)
    save_vocab(vocab, filename)
    return vocab


def load_data(filename, header, y_idx, numer_features, categ_features, shuffle=False, delim=','):
    if categ_features is None:
        categ_features = []

    if numer_features is None:
        numer_features = []

    df = pd.read_csv(filename, header=None, skiprows=(1 if header else None), delimiter=delim)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    labels = df[y_idx].values
    data = df[numer_features].copy()
    data = data.apply(pd.to_numeric)
    data.columns = range(data.shape[1])
    n = len(numer_features)
    for col in data:
        mean = data[col].mean()
        std = data[col].std()
        # apply standard scaler and set missing values to mean
        data[col] = data[col].apply(lambda x: 0 if math.isnan(x) else (x - mean) / std)

    print('Size:', len(data))
    for i in categ_features:
        vocab = get_vocab(n, df[i].fillna(UNK_TOKEN))
        data[n] = onehot_encode(df[i], vocab)
        n += 1

    # Family size feature
    # data[n] = df[6] + df[7] + 1
    # mean = data[n].mean()
    # std = data[n].std()
    # data[n] = data[n].apply(lambda x: 0 if math.isnan(x) else (x - mean) / std)
    # n += 1

    # Title feature
    # titles = df[3].apply(get_title)
    # vocab = get_vocab(n, titles)
    # data[n] = onehot_encode(titles, vocab)
    # n += 1

    # Age bin feature
    # mean = df[5].mean()
    # age_bins = pd.cut(df[5].fillna(mean), bins=[0, 12, 20, 40, 120], labels=['Child', 'Teen', 'Young Adult', 'Adult'])
    # vocab = get_vocab(n, age_bins)
    # data[n] = onehot_encode(age_bins, vocab)
    # n += 1

    # Fare bin feature
    # fares = df[9].copy()
    # mean = fares.mean()
    # print('Mean fare:', mean)
    # fares.fillna(mean, inplace=True)
    # fares.loc[fares == 0] = mean
    # fare_bins = pd.cut(fares, bins=[0, 7.91, 14.45, 31, 600], labels=['Low', 'Median', 'Mean', 'High'])
    # vocab = get_vocab(n, fare_bins)
    # data[n] = onehot_encode(fare_bins, vocab)
    # n += 1

    print(data.shape)
    return data, labels, df


def load_vocab(filename):
    vocab = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            token = line.rstrip()
            if token:
                vocab[token] = i

    return vocab


def onehot_encode(values, vocab):
    vocab_size = len(vocab)
    m = len(values)
    encoded = np.zeros((m, vocab_size))
    unk_i = vocab[UNK_TOKEN]
    for i, val in enumerate(values):
        encoded[i, vocab.get(str(val), unk_i)] = 1

    return list(encoded)


def save_train_test_split(data, test_ratio, train_filename, test_filename):
    data_size = len(data)
    test_size = int(data_size * test_ratio)
    test_set = data[:test_size]
    train_set = data[test_size:]
    train_set.to_csv(train_filename, header=False, index=False)
    test_set.to_csv(test_filename, header=False, index=False)


def save_vocab(vocab, filename):
    with open(filename, 'w') as f:
        for key, val in sorted(vocab.items(), key=itemgetter(1)):
            f.write(key + '\n')


def train_test_split(data, labels, test_ratio):
    data_size = len(data)
    test_size = int(data_size * test_ratio)
    x_test = data[:test_size]
    y_test = labels[:test_size]
    x_train = data[test_size:]
    y_train = labels[test_size:]
    return x_train, y_train, x_test, y_test
