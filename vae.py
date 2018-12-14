import tensorflow as tf
import tensorflow_probability.python.distributions as tfd


class Vae(object):
    def __init__(self, latent_size, activation='relu', analytic_kl=True, mixture_components=1, n_samples=16):
        self.latent_size = latent_size
        self.activation = activation
        self.analytic_kl = analytic_kl
        self.mixture_components = mixture_components
        self.n_samples = n_samples

    def _make_encoder(self, latent_size, activation):
        raise NotImplementedError("Subclasses should implement this!")

    def _make_decoder(self, activation):
        raise NotImplementedError("Subclasses should implement this!")

    def loss(self, features):
        encoder = self._make_encoder(self.latent_size, self.activation)
        decoder = self._make_decoder(self.latent_size, self.activation)
        latent_prior = self._make_prior()

        approx_posterior = encoder(features)
        approx_posterior_sample = approx_posterior.sample(self.n_samples)
        decoder_likelihood = decoder(approx_posterior_sample)
        distortion = -decoder_likelihood.log_prob(features)
        if self.analytic_kl:
            rate = tfd.kl_divergence(approx_posterior, latent_prior)
        else:
            rate = (approx_posterior.log_prob(approx_posterior_sample)
                    - latent_prior.log_prob(approx_posterior_sample))
        elbo_local = -(rate + distortion)

        elbo = tf.reduce_mean(elbo_local)
        loss = -elbo
        return loss

    def _make_prior(self):
        if self.mixture_components == 1:
            # See the module docstring for why we don't learn the parameters here.
            return tfd.MultivariateNormalDiag(
                loc=tf.zeros([self.latent_size]),
                scale_identity_multiplier=1.0)

        loc = tf.get_variable(name="loc", shape=[self.mixture_components, self.latent_size])
        raw_scale_diag = tf.get_variable(
            name="raw_scale_diag", shape=[self.mixture_components, self.latent_size])
        mixture_logits = tf.get_variable(
            name="mixture_logits", shape=[self.mixture_components])

        return tfd.MixtureSameFamily(
            components_distribution=tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=tf.nn.softplus(raw_scale_diag)),
            mixture_distribution=tfd.Categorical(logits=mixture_logits),
            name="prior")
