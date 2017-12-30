"""Quick tests of the binary neural networks.."""

import numpy as np
from bnn import BinaryNeuralNetwork


def test_bnn():
    """Test a binary neural network."""
    in_dim = (28, 28, 1)
    out_dim = 10
    n_imgs = 100
    bnn = BinaryNeuralNetwork(in_dim=in_dim, out_dim=out_dim)

    x = np.random.randn(n_imgs, *in_dim)
    y = np.random.randint(out_dim, size=n_imgs)

    bnn.fit(x, y, x, y)
    vals = bnn.inspect_network(x, y)
    param_file = bnn.param_file
    bnn = BinaryNeuralNetwork(in_dim=in_dim, out_dim=out_dim,
                              param_file=param_file)
    vals1 = bnn.inspect_network(x, y)
    assert np.array_equal(vals['score'], vals1['score'])


def test_tnn():
    """Test a ternary neural network."""
    network_params = {
        'conv': [32],
        'max_pool': [True, False],
        'filter_size': 3,
        'fc': [10],
        'bin_weights': [True] * 1,
        'bin_acts': [True] * 1,
        'levels': 3,
        'threshold': 0.2,
    }
    in_dim = (28, 28, 1)
    out_dim = 10
    n_imgs = 100
    bnn = BinaryNeuralNetwork(in_dim=in_dim, out_dim=out_dim,
                              network_params=network_params)

    x = np.random.randn(n_imgs, *in_dim)
    y = np.random.randint(out_dim, size=n_imgs)

    bnn.fit(x, y, x, y)
    vals = bnn.inspect_network(x, y)
    param_file = bnn.param_file
    bnn = BinaryNeuralNetwork(in_dim=in_dim, out_dim=out_dim,
                              param_file=param_file,
                              network_params=network_params)
    vals1 = bnn.inspect_network(x, y)
    assert np.array_equal(vals['score'], vals1['score'])


if __name__ == '__main__':
    test_bnn()
    test_tnn()
