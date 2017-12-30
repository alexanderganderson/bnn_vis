"""Operations on graphs."""
import abc
import os
import numpy as np
import tensorflow as tf


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


class DataGenerator(object):
    """Abstract class for an object that creates generators for iterating
    through the data."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_generator(self, batch_size=None):
        """
        Get a generator that iterates over the data.

        Parameters
        ----------
        batch_size : int
            Batch size

        Returns
        -------
        gen : python generator
            Generator that yields data in batches
        """
        pass

    @abc.abstractproperty
    def data_size(self):
        """Return the number of examples in the data."""
        pass


def _session_config():
    """Generate config proto for session."""
    config = tf.ConfigProto()
    # Only use GPU memory necessary
    config.gpu_options.allow_growth = True
    return config


def _train_graph(
    graph,
    init_op,
    train_op,
    valid_dict,
    saver,
    train_data_object,
    valid_data_object,
    feed_vars,
    batch_size,
    num_epochs,
    param_file,
    checkpoint_every_n_epochs,
    global_step,
    save_path='tmp/model.ckpt'
):
    """
    Train a computational graph.

    Parameters
    ----------
    graph : tf.Graph
        Computational graph
    init_op : tf.op
        Op to initialize the graph
    train_op : tf.op
        Op to run one training step of the network
    valid_dict : dict of tf.op
        Dictionary of ops to run for validation
    saver : tf.Saver
       Object to save variables
    train_data_object : DataGenerator
        Object to create training data generators
    valid_data_object : DataGenerator
        Object to create validation data generators
    feed_vars : dict of tf.op
        Dict of ops for inputs to the network
    batch_size : int
        Batch size
    num_epochs : int
        Number of epochs to train
    param_file : str
        If not None, restore graph from here
    checkpoint_every_n_epochs :
    global_step : tf.Op
        Op tracking global step
    save_path : str
        Base of path to save model

    Returns
    -------
    new_save_path : str
        Path that the model was saved to.
    """
    config = _session_config()
    inc_global_step = tf.assign(global_step, global_step + 1)
    with tf.Session(graph=graph, config=config) as sess:
        if param_file is None:
            sess.run(init_op)
        else:
            saver.restore(sess, param_file)

        make_dir(os.path.dirname(save_path))
        for j in range(num_epochs):
            sess.run(inc_global_step)

            train_data_generator = train_data_object.get_generator(batch_size)
            for feed_values in train_data_generator:
                sess.run(
                    train_op,
                    feed_dict=dict(zip(feed_vars, feed_values))
                )

            valid_data_generator = valid_data_object.get_generator()
            for feed_values in valid_data_generator:
                summary_vals = sess.run(
                    valid_dict, feed_dict=dict(zip(feed_vars, feed_values))
                )
                print(''.join(['{}: {:0.6f} '.format(k, v) for k, v in
                               summary_vals.items()]))

            if j % checkpoint_every_n_epochs == 0:
                new_save_path = saver.save(
                    sess, save_path, global_step=j)

        new_save_path = saver.save(sess, save_path)
    return new_save_path


def _merge_list_of_dict(l):
    assert len(l) > 0
    keys = l[0].keys()
    res = {}
    for k in keys:
        tmp = []
        for d in l:
            tmp.append(d[k])
        res[k] = np.concatenate(tmp, axis=0)

    return res


def _eval_graph(
    graph,
    saver,
    eval_ops,
    data_object,
    feed_vars,
    param_file,
    batch_size=50,
):
    """
    Evaluate a computational graph.

    Parameters
    ----------
    graph : tf.Graph
        Computational graph
    saver : tf.Saver
        Object to save and restore graph.
    eval_ops : dict of tf.op
        Ops to evaluate.
    data_object : DataGenerator
        Input data for evaluation
    feed_vars : list of ops
        Ops for input.
    param_file : str
        Path to restore file.

    Returns
    -------
    eval_vals : list of arrays
        Values of eval ops run on data.
    """
    assert isinstance(eval_ops, dict)
    config = _session_config()
    with tf.Session(graph=graph, config=config) as sess:
        saver.restore(sess, param_file)
        data_generator = data_object.get_generator()
        eval_vals_ = []
        for feed_values in data_generator:
            eval_vals = sess.run(
                eval_ops, feed_dict=dict(zip(feed_vars, feed_values)))
            eval_vals_.append(eval_vals)
    return _merge_list_of_dict(eval_vals_)
