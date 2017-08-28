#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Writen by Lianghong  2017-08-28 11:19:01

import mxnet as mx
import mxnet.gluon as gluon
import os
import gzip
import struct
import warnings
import numpy as np

class _DownloadedDataset(gluon.data.Dataset):
    """Base class for MNIST, cifar10, etc."""
    def __init__(self, root, train, transform):
        self._root = os.path.expanduser(root)
        self._train = train
        self._transform = transform
        self._data = None
        self._label = None

        self._get_data()

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx], self._label[idx])
        return self._data[idx], self._label[idx]

    def __len__(self):
        return len(self._label)

    def _get_data(self):
        raise NotImplementedError

class FASHION_MNIST(_DownloadedDataset):
    def __init__(self, root='~/.mxnet/datasets/', train=True,
                 transform=None):
        super(FASHION_MNIST, self).__init__(root, train, transform)

    def _get_data(self):
        if self._train:
            data_file = "./data/train-images-idx3-ubyte.gz"
            label_file = "./data/train-labels-idx1-ubyte.gz"
        else:
            data_file = "./data/t10k-images-idx3-ubyte.gz"
            label_file = "./data/t10k-labels-idx1-ubyte.gz"

        with gzip.open(label_file, 'rb') as fin:
            struct.unpack(">II", fin.read(8))
            label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)

        with gzip.open(data_file, 'rb') as fin:
            struct.unpack(">IIII", fin.read(16))
            data = np.fromstring(fin.read(), dtype=np.uint8)
            data = data.reshape(len(label), 28, 28, 1)

        self._data = [mx.nd.array(x, dtype=x.dtype) for x in data]
        self._label = label



