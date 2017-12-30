"""Directories containing data."""
import os
import numpy as np
import gzip
import cPickle as pkl

MNIST = 'MNIST'
CIFAR10 = 'cifar-10'

MNIST_DATA_PATH = '/home/cberg/mnist'
CIFAR10_DATA_PATH = '/home/aga/cifar10/cifar-10-batches-py'


def unpickle(fn):
    with open(fn, 'rb') as f:
        d = pkl.load(f)
    return d


def load_data(dataset):
    if dataset == MNIST:
        return load_mnist()
    elif dataset == CIFAR10:
        return load_cifar10()
    else:
        raise ValueError('Invalid Dataset')

def load_mnist(data_path=MNIST_DATA_PATH):
    """
    Returns
    -------
    train_images: array, shape (50000, 784)
    train_labels: array, shape (50000)
    test_images: array, shape (10000, 784)
    test_labels: array, shape (10000)
    cv_images: array, shape (10000, 784)
    cv_labels: array, shape (10000)
    """
    # FIXME: hard code filenames
    with gzip.open(os.path.join(data_path, 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pkl.load(f)

    train_images = train_set[0].reshape([-1, 28, 28, 1])
    train_labels = train_set[1]

    cv_images = valid_set[0].reshape([-1, 28, 28, 1])
    cv_labels = valid_set[1]

    test_images = test_set[0].reshape([-1, 28, 28, 1])
    test_labels = test_set[1]

    return (train_images, train_labels,
            cv_images, cv_labels,
            test_images, test_labels)


def load_cifar10(data_dir=CIFAR10_DATA_PATH, train=True, test=True):
    """
    Loads cifar-10 images and sparse labels into a training set of 50000
    and a test set of 10000

    Parameters
    ----------
    data_dir: absolute path to directory containing cifar-10 files

    Returns
    -------
    train_images: array, shape (50000, 32 ,32 ,3)
    train_labels: array, shape (50000)
    test_images: array, shape (10000, 32, 32, 3)
    test_labels: array, shape (10000)
    """

    train_files = [fn for fn in os.listdir(data_dir) if fn.startswith('data')]

    test_files = [fn for fn in os.listdir(data_dir) if fn.startswith('test')]

    test_files.sort()
    train_files.sort()

    l_i = 32
    if train:
        for i in range(len(train_files)):
            datum = unpickle(os.path.join(data_dir, train_files[i]))
            image = datum['data'].reshape(-1, 3, l_i, l_i).transpose(0, 2, 3, 1)
            label = datum['labels']
            if i == 0:
                train_images = image
                train_labels = label
            else:
                train_images = np.concatenate((train_images, image), axis=0)
                train_labels = np.concatenate((train_labels, label))
                #FIXME: inefficient

        train_images = train_images * 2. / 255 - 1

        cv_images = train_images[-500:]
        cv_labels = train_labels[-500:]

        train_images = train_images[:-500]
        train_labels = train_labels[:-500]

    else:
        train_images, train_labels = None, None
        cv_images, cv_labels = None, None

    if test:
        datum = unpickle(os.path.join(data_dir, test_files[0]))
        test_images = datum['data'].reshape(-1, 3, l_i, l_i).transpose(0, 2, 3, 1)
        test_labels = np.array(datum['labels'])
        test_images = test_images * 2. / 255 - 1
    else:
        test_images, test_labels = None, None

    return (train_images, train_labels,
            cv_images, cv_labels,
            test_images, test_labels)
