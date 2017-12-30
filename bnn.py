from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from utils import clip, flatten_conv, add_binary_layer, ClassifierGenerator
from graph_ops import _train_graph, _eval_graph


class BinaryNeuralNetwork(object):
    """
    Creates a neural network with binary weights and activations.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        network_params={
            'conv': [32],
            'filter_size': 3,
            'max_pool': [True],
            'fc': [10],
            'bin_weights': [True] * 1,
            'bin_acts': [True] * 1,
            'levels': 2,
            'threshold': 0,
        },
        optimizer_params={
            'batch_size': 10,
            'num_epochs': 4,
            'method': 'adam',
            'learning_rate': 0.001,
            'epoch_decay_rate': 0.95,
            'loss_mode': 'hinge-loss',
            'ckpt_n': 2,
            'save_path': 'tmp/model.ckpt'
        },
        param_file=None
    ):
        """


        Parameters
        ----------
        in_dim : tuple
            Dimensions of the input data
        out_dim : int
            Dimensions of output
        network_params : dict
            'conv' : list of int
                Number of feature maps for each conv layer
            'filter_size' : int
                Receptive field size of each conv layer
            'max_pool' : list of bool
                Whether to max pool after each conv layer
            'fc': list of int
                Number of hidden units in each fc layer (excluding last layer)
            'bin_weights' : list of bool
                Whether to binarize weights for each conv layer
            'bin_acts' : list of bool
                Whether to binarize activations for each conv layer
            'levels' : int
                2: binary, 3: ternary
            'threshold' : float
                Threshold for ternarization function
        optimizer_params : dict
            'num_epochs' : int
                Number of training epochs
            'method' : str
                Gradient descent method.
            'learning_rate' : float
                Initial learning rate.
            'epoch_decay_rate' :
                Learning rate decay factor for each epoch.
            'loss_mode' : str
                Softmax or hinge loss
            'ckpt_n': int
                Checkpoint very n iterations
            'save_path': str
                Path to save network
        param_file : str
            Path to restore network (if None, initialize new network)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.network_params = network_params
        self.optimizer_params = optimizer_params
        self.param_file = param_file

    def _build_graph(self, training=True):
        """Build the tensorflow graph in either training or validation mode."""
        g = tf.Graph()

        conv_ = self.network_params['conv']
        filter_size = self.network_params['filter_size']
        max_pool = self.network_params['max_pool']
        fc_ = self.network_params['fc']
        bin_weights = self.network_params['bin_weights']
        bin_act = self.network_params['bin_acts']
        d_out = self.out_dim
        loss_mode = self.optimizer_params['loss_mode']
        levels = self.network_params['levels']
        threshold = self.network_params['threshold']

        with g.as_default():
            with tf.variable_scope('inputs'):
                x = tf.placeholder(tf.float32, [None] + list(self.in_dim))
            with tf.variable_scope('targets'):
                y = tf.placeholder(tf.int32, [None])
                # must be int32 for sparse to one_hot conversion

            top = x

            # Convolutional Layers
            conv_ = [self.in_dim[-1]] + conv_
            conv_layers = len(conv_) - 1

            conv_tensors = []
            for i in range(conv_layers):
                top, tensor_dict = add_binary_layer(
                    top,
                    'layer_{}'.format(i+1),
                    filter_shape=(filter_size, filter_size,
                                  conv_[i], conv_[i+1]),
                    max_pool=max_pool[i],
                    bin_weights=bin_weights[i],
                    bin_act=bin_act[i],
                    levels=levels,
                    threshold=threshold,
                    training=training,
                )
                conv_tensors.append(tensor_dict)

            # Fully Connected Layers
            top, dim = flatten_conv(top)

            fc_ = [dim] + fc_ + [d_out]
            fc_layers = len(fc_) - 1

            fc_tensors = []
            for i in range(fc_layers):
                bin_act = True
                bin_weights = True
                # FIXME: Terrible
                bin_act1 = (i < fc_layers - 1) and bin_act
                top, tensor_dict = add_binary_layer(
                    top,
                    'fc_layer_{}'.format(i+1),
                    (fc_[i], fc_[i+1]),
                    bin_act=bin_act1,
                    bin_weights=bin_weights,
                    levels=levels,
                    threshold=threshold,
                    training=training,
                )
                fc_tensors.append(tensor_dict)

            if loss_mode == 'softmax':
                y_oh = tf.one_hot(y, d_out)
                y_ = tf.nn.softmax(top)
                cost = tf.reduce_mean(-tf.reduce_sum(y_oh*tf.log(y_),
                                                     reduction_indices=[1]))
            elif loss_mode == 'hinge-loss':
                y_oh = tf.one_hot(y, d_out, off_value=-1.)
                y_ = top
                cost = tf.reduce_mean(tf.square(tf.maximum(0., 1. - y_*y_oh)))
            else:
                raise ValueError('Invalid loss_mode {}'.format(loss_mode))

            correct = tf.equal(tf.argmax(y_oh, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            init_op = tf.global_variables_initializer()

            saver = tf.train.Saver()

            d = {
                'graph': g,
                'cost': cost,
                'subcosts': {},
                'feed_vars': [x, y],
                'conv_tensors': conv_tensors,
                'fc_tensors': fc_tensors,
                'init_op': init_op,
                'saver': saver,
                'score': correct,
                'summary_ops': OrderedDict((
                    ('cost', cost),
                    ('accuracy', accuracy)
                )),
            }

            if not training:
                return d

            # FIXME: brittle
            bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # Training ops
            global_step = tf.Variable(
                0, name='global_step', trainable=False, dtype=tf.int32)
            with tf.variable_scope('learning_rate'):
                lr = tf.train.exponential_decay(
                    self.optimizer_params['learning_rate'],
                    global_step,
                    1,
                    self.optimizer_params['epoch_decay_rate'],
                    staircase=True
                )

            if self.optimizer_params['method'] == 'adam':
                opt = tf.train.AdamOptimizer(lr)
            else:
                raise ValueError('Invalid optimizer method')

            gvs = opt.compute_gradients(cost)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                          for grad, var in gvs]
            train_op = opt.apply_gradients(capped_gvs)

            weights = [var for var in tf.trainable_variables()
                       if 'W' in var.name]

            clip_ops = [tf.assign(weight, clip(weight)) for weight in weights]

            train_op = control_flow_ops.group(
                train_op, *(bn_update_ops + clip_ops),
                name='train_with_clip')

            init_op = tf.global_variables_initializer()

            d['bn_update_ops'] = bn_update_ops,
            d['init_op'] = init_op
            d['train_op'] = train_op
            d['summary_ops']['lr'] = lr
            d['summary_ops']['global_step'] = global_step

            d['global_step'] = global_step
            return d

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Fit the BNN to data.

        Parameters
        ----------
        x_train : array, shape (n_samples, ...)
            Network inputs for training.
        y_train : array, shape (n_samples, ...)
            Regression targets for training data.
        x_valid : array, shape (n_samples, ...)
            Network inputs for validation.
        y_valid : array, shape (n_samples, ...)
            Regression targets for training.

        Returns
        -------
        self : BinaryNeuralNetwork
            Trained network.
        """
        train_obj, valid_obj = [
            ClassifierGenerator(images, labels) for images, labels in [
                (x_train, y_train),
                (x_valid, y_valid),
            ]
        ]

        d = self._build_graph(training=True)
        param_file = _train_graph(
            graph=d['graph'],
            init_op=d['init_op'],
            train_op=d['train_op'],
            valid_dict=d['summary_ops'],
            saver=d['saver'],
            train_data_object=train_obj,
            valid_data_object=valid_obj,
            feed_vars=d['feed_vars'],
            batch_size=self.optimizer_params['batch_size'],
            num_epochs=self.optimizer_params['num_epochs'],
            checkpoint_every_n_epochs=self.optimizer_params['ckpt_n'],
            global_step=d['global_step'],
            param_file=self.param_file,
            save_path=self.optimizer_params['save_path'])
        self.param_file = param_file
        return self

    def score(self, x, y):
        return None

    def refit_batch_norm(self, x):
        pass

    def inspect_network(self, x, y):
        data_object = ClassifierGenerator(x, y)
        d = self._build_graph(training=False)

        eval_ops = {
            'score': d['score'],
            'conv_tensors': d['conv_tensors'],
            'fc_tensors': d['fc_tensors'],
            'x': d['feed_vars'][0]
        }

        eval_vals = _eval_graph(
            graph=d['graph'],
            saver=d['saver'],
            eval_ops=eval_ops,
            data_object=data_object,
            feed_vars=d['feed_vars'],
            param_file=self.param_file
        )
        return eval_vals

    def classify(self, x, y):
        data_object = ClassifierGenerator(x, y)
        d = self._build_graph(training=False)

        eval_ops = {'score': d['score']}

        eval_vals = _eval_graph(
            graph=d['graph'],
            saver=d['saver'],
            eval_ops=eval_ops,
            data_object=data_object,
            feed_vars=d['feed_vars'],
            param_file=self.param_file
        )
        return eval_vals
