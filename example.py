import os
import urllib
from absl import flags
import numpy as np
import tensorflow as tf
from cnn_vae import CnnVae

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "max_steps", default=5001, help="Number of training steps to run.")
flags.DEFINE_integer(
    "latent_size",
    default=16,
    help="Number of dimensions in the latent code (z).")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_integer(
    "n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_integer(
    "mixture_components",
    default=100,
    help="Number of mixture components to use in the prior. Each component is "
         "a diagonal normal distribution. The parameters of the components are "
         "intialized randomly, and then learned along with the rest of the "
         "parameters. If `analytic_kl` is True, `mixture_components` must be "
         "set to `1`.")
flags.DEFINE_bool(
    "analytic_kl",
    default=False,
    help="Whether or not to use the analytic version of the KL. When set to "
         "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
         "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
         "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
         "then you must also specify `mixture_components=1`.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=500, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")

FLAGS = flags.FLAGS

ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


def download(directory, filename):
    """Downloads a file."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    url = os.path.join(ROOT_PATH, filename)
    print("Downloading %s to %s" % (url, filepath))
    urllib.request.urlretrieve(url, filepath)
    return filepath


def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


def static_mnist_dataset(directory, split_name):
    """Returns binary static MNIST tf.data.Dataset."""
    amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
    dataset = tf.data.TextLineDataset(amat_file)
    str_to_arr = lambda string: np.array([c == b"1" for c in string.split()])

    def _parser(s):
        booltensor = tf.py_func(str_to_arr, [s], tf.bool)
        reshaped = tf.reshape(booltensor, [28, 28, 1])
        return tf.to_float(reshaped), tf.constant(0, tf.int32)

    return dataset.map(_parser)


def build_input_fns(data_dir, batch_size):
    """Builds an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = static_mnist_dataset(data_dir, "train")
    training_dataset = training_dataset.shuffle(50000).repeat().batch(batch_size)
    train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

    # Build an iterator over the heldout set.
    eval_dataset = static_mnist_dataset(data_dir, "valid")
    eval_dataset = eval_dataset.batch(batch_size)
    eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

    return train_input_fn, eval_input_fn


def model_fn(features, labels, mode, params, config):
    del labels, config

    if params["analytic_kl"] and params["mixture_components"] != 1:
        raise NotImplementedError(
            "Using `analytic_kl` is only supported when `mixture_components = 1` "
            "since there's no closed form otherwise.")

    model = CnnVae(latent_size=params['latent_size'],
                   activation=params['activation'],
                   analytic_kl=params['analytic_kl'],
                   mixture_components=params['mixture_components'],
                   n_samples=params['n_samples'],
                   base_depth=params['base_depth'],
                   output_shape=IMAGE_SHAPE)
    loss = model.loss(features)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                          params["max_steps"])
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
        },
    )


def main(argv):
    del argv

    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    train_input_fn, eval_input_fn = build_input_fns(FLAGS.data_dir,
                                                    FLAGS.batch_size)

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_steps=FLAGS.viz_steps,
        ),
    )

    for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
        estimator.train(train_input_fn, steps=FLAGS.viz_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == "__main__":
    tf.app.run()
