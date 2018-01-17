"""Train a Binary Neural Network."""

from argparse import ArgumentParser
from data.load_data import load_data, MNIST, CIFAR10
from bnn import BinaryNeuralNetwork


def setup(argv=None):
    """
    Parse command line arguments to pass into the rest of the code.

    Parameters specify:
        The dataset.
        The architecture of a neural network with multiple
            convolutional layers then multiple fully-connected layers.
        Optimization Parameters.
        Debug mode (regression test for small dataset).


    Parameters
    ----------
    argv : list of str
        If None, just uses sys.argv, else, pass in the corresponding string
            of command line arguments split at spaces
    """

    parser = ArgumentParser('Trains a BNN on a chosen dataset')

    parser.add_argument('--param_file', type=str, default=None,
                        help="file containing saved model if it exists")

    parser.add_argument('--data_set', type=str, default=CIFAR10,
                        help="data set to train")

    parser.add_argument('--batch_size', type=int, default=50,
                        help="size of batches for training")

    parser.add_argument('--epochs', type=int, default=500,
                        help="number of epochs in training")

    parser.add_argument('--lr_start', type=float, default=0.001,
                        help="learning rate for start of training")

    parser.add_argument('--lr_end', type=float, default=0.0000003,
                        help="learning rate at which to end training")

    parser.add_argument('--loss_mode', type=str, default='hinge-loss',
                        help="loss function to use when training")

    parser.add_argument('--conv', nargs='*', type=int,
                        default=[128, 128, 256, 256, 512, 512],
                        help="number of units in each convolutional layer")

    parser.add_argument('--fc', nargs='*', default=[1024, 1024],
                        type=int,
                        help="number of units in each fully connected layer")

    parser.add_argument('--max_pool', nargs='*', type=int,
                        default=[0, 1, 0, 1, 0, 1],
                        help="T/F for max pooling for each layer")

    parser.add_argument('--ckpt_n', type=int, default=10,
                        help="checkpoint every n epochs")

    parser.add_argument('--filter_size', type=int, default=3,
                        help="size of filters used for convolution")

    parser.add_argument('--continuous', action='store_false', dest='binary',
                        default=True,
                        help="make weights continuous")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="debug mode")

    parser.add_argument('--levels', type=int, default=2,
                        help="number of levels to quantize to")

    parser.add_argument('--threshold', type=float, default=0.0,
                        help="threshold for quantization")

    args = parser.parse_args(argv)
    assert len(args.max_pool) == len(args.conv)

    network_params = {
        'conv': args.conv,
        'max_pool': args.max_pool,
        'filter_size': args.filter_size,
        'fc': args.fc,
        'bin_weights': [args.binary] * len(args.conv),
        'bin_acts': [args.binary] * len(args.conv),
        'levels': args.levels,
        'threshold': args.threshold,
    }

    optimizer_params = {
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'train_size': None,
        'method': 'adam',
        'learning_rate': args.lr_start,
        'epoch_decay_rate': (args.lr_end/args.lr_start)**(1./args.epochs),
        'loss_mode': args.loss_mode,
        'ckpt_n': args.ckpt_n,
        'save_path': 'tmp/{}_model_{}_levels1.ckpt'.format(
            args.data_set, args.levels)
    }

    return (args.param_file, args.data_set, network_params, optimizer_params,
            args.debug)


def main():
    param_file, data_set, network_params, optimizer_params, debug = setup()

    if data_set == CIFAR10:
        (train_images, train_labels,
         cv_images, cv_labels,
         test_images, test_labels) = load_data(CIFAR10)

        if debug:
            train_images = train_images[0:100]
            train_labels = train_labels[0:100]

        in_dim = (32, 32, 3)
        d_out = 10

    elif data_set == MNIST:
        (train_images, train_labels,
         cv_images, cv_labels,
         test_images, test_labels) = load_data(MNIST)
        in_dim = (28, 28, 1)
        d_out = 10

    else:
        raise ValueError('Invalid data_set {}'.format(data_set))

    bnn = BinaryNeuralNetwork(
        in_dim=in_dim,
        out_dim=d_out,
        network_params=network_params,
        optimizer_params=optimizer_params,
        param_file=param_file
    )

    bnn.fit(train_images, train_labels, cv_images, cv_labels)


if __name__ == '__main__':
    main()
