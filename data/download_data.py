"""Download and extract MNIST and/or CIFAR10."""

import os
from argparse import ArgumentParser

MNIST = 'MNIST'
CIFAR10 = 'cifar-10'

# FIXME: extracts again even if already extracted
# FIXME: no clean function

def mnist():
    """Download and extract MNIST."""

    MNIST_DATA_PATH = 'mnist'
    MNIST_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    MNIST_FN = 'mnist.pkl.gz'

    save_path = os.path.join(MNIST_DATA_PATH, MNIST_FN)

    if not os.path.exists(save_path):
        print 'Downloading file'
        # -P specifies directory to save to
        os.system('wget -P {} {}'.format(MNIST_DATA_PATH, MNIST_URL))
    if os.path.exists(save_path):
        print 'Uncompressing mnist archive file'
        # -k keeps original archive file
        os.system('gunzip -k {}'.format(save_path))



def cifar10():
    """Download and extract CIFAR10."""
    CIFAR10_DATA_PATH = 'cifar10'
    CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    CIFAR10_FN = 'cifar-10-python.tar.gz'

    save_path = os.path.join(CIFAR10_DATA_PATH, CIFAR10_FN)

    if not os.path.exists(save_path):
        print 'Downloading file'
        os.system('wget -P {} {}'.format(CIFAR10_DATA_PATH, CIFAR10_URL))

    if os.path.exists(save_path):
        # -C gives destination directory
        os.system('tar -xzf {} -C {}'.format(save_path, CIFAR10_DATA_PATH))


if __name__ == "__main__":
    parser = ArgumentParser('Download a dataset')
    parser.add_argument('--data_set', type=str,
                        help="data set to train")
    data_set = parser.parse_args().data_set
    if data_set == MNIST:
        mnist()
    elif data_set == CIFAR10:
        cifar10()
