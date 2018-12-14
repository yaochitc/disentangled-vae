import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from tensorflow.keras.layers import Dense
from vae import Vae


class DnnVae(Vae):
    def __init__(
            self,
            output_shape,
            **kwargs):
        super(DnnVae, self).__init__(**kwargs)
        self.output_shape = output_shape

    def _softplus_inverse(self, x):
        """Helper which computes the function inverse of `tf.nn.softplus`."""
        return tf.log(tf.math.expm1(x))

    def _make_encoder(self, latent_size, activation):
        encoder_net = tf.keras.Sequential([
            Dense(256, activation=activation),
            Dense(64, activation=activation),
            Dense(2 * latent_size, activation=None),
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
        decoder_net = tf.keras.Sequential([
            Dense(64, activation=activation),
            Dense(256, activation=activation),
            Dense(self.output_shape, activation=None),
        ])

        def decoder(codes):
            logits = decoder_net(codes)
            logits_logstd = tf.get_variable('logits_logstd', shape=[],
                                            initializer=tf.constant_initializer(0.))
            return tfd.Independent(tfd.Normal(loc=logits, scale=tf.exp(logits_logstd)),
                                   reinterpreted_batch_ndims=1,
                                   name="logits")

        return decoder
