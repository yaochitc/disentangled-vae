import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import functools
from vae import Vae


class CnnVae(Vae):
    def __init__(
            self,
            base_depth,
            output_shape,
            **kwargs):
        super(CnnVae, self).__init__(**kwargs)
        self.base_depth = base_depth
        self.output_shape = output_shape

    def _softplus_inverse(self, x):
        """Helper which computes the function inverse of `tf.nn.softplus`."""
        return tf.log(tf.math.expm1(x))

    def _make_encoder(self, latent_size, activation):
        conv = functools.partial(
            tf.keras.layers.Conv2D, padding="SAME", activation=activation)

        encoder_net = tf.keras.Sequential([
            conv(self.base_depth, 5, 1),
            conv(self.base_depth, 5, 2),
            conv(2 * self.base_depth, 5, 1),
            conv(2 * self.base_depth, 5, 2),
            conv(4 * latent_size, 7, padding="VALID"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2 * latent_size, activation=None),
        ])

        def encoder(images):
            images = 2 * tf.cast(images, dtype=tf.float32) - 1
            net = encoder_net(images)
            return tfd.MultivariateNormalDiag(
                loc=net[..., :latent_size],
                scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                          self._softplus_inverse(1.0)),
                name="code")

        return encoder

    def _make_decoder(self, latent_size, activation):
        deconv = functools.partial(
            tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
        conv = functools.partial(
            tf.keras.layers.Conv2D, padding="SAME", activation=activation)

        decoder_net = tf.keras.Sequential([
            deconv(2 * self.base_depth, 7, padding="VALID"),
            deconv(2 * self.base_depth, 5),
            deconv(2 * self.base_depth, 5, 2),
            deconv(self.base_depth, 5),
            deconv(self.base_depth, 5, 2),
            deconv(self.base_depth, 5),
            conv(self.output_shape[-1], 5, activation=None),
        ])

        def decoder(codes):
            original_shape = tf.shape(codes)
            # Collapse the sample and batch dimension and convert to rank-4 tensor for
            # use with a convolutional decoder network.
            codes = tf.reshape(codes, (-1, 1, 1, latent_size))
            logits = decoder_net(codes)
            logits = tf.reshape(
                logits, shape=tf.concat([original_shape[:-1], self.output_shape], axis=0))
            return tfd.Independent(tfd.Bernoulli(logits=logits),
                                   reinterpreted_batch_ndims=len(self.output_shape),
                                   name="image")

        return decoder
