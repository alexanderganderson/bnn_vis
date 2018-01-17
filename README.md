# The High-Dimensional Geometry of Binary Neural Networks

This repository contains code to reproduce the results in the following
paper:

[The High-Dimensional Geometry of Binary Neural Networks](https://arxiv.org/abs/1705.07199)

Alexander G. Anderson, Cory P. Berg


**Abstract**

Recent research has shown that one can train a neural network with binary
weights and activations at train time by augmenting the weights with a
high-precision continuous latent variable that accumulates small changes from
stochastic gradient descent. However, there is a dearth of work to explain why
one can effectively capture the features in data with binary weights and
activations. Our main result is that the neural networks with binary weights
and activations trained using the method of Courbariaux, Hubara et al. (2016)
work because of the high-dimensional geometry of binary vectors. In particular,
the ideal continuous vectors that extract out features in the intermediate
representations of these BNNs are well-approximated by binary vectors in the
sense that dot products are approximately preserved. Compared to previous
research that demonstrated good classification performance with BNNs, our work
explains why these BNNs work in terms of HD geometry.  Furthermore, the results
and analysis used on BNNs are shown to generalize to neural networks with
ternary weights and activations. Our theory serves as a foundation for
understanding not only BNNs but a variety of methods that seek to compress
traditional neural networks. Furthermore, a better understanding of multilayer
binary neural networks serves as a starting point for generalizing BNNs to
other neural network architectures such as recurrent neural networks.


## Quickstart:

1. Download the relevant datasets. From the data directory, run:

`python download_data.py --data_set MNIST`

`python download_data.py --data_set cifar-10`

2. Run a quick test: `python bnn_test.py`

3. Train a network: `python run_bnn.py`

4. Visualize a trained network: `jupyter notebook gen_plots.ipynb`

5. Analyze quantization of HD vectors assuming a Gaussian distribution:
`juypter notebook hd_figures.ipynb`

Be sure to choose the appropriate path for loading the model.

[Note if the parameters are changed in `run_bnn.py`, they need to be changed in
the visualization script as well.]
