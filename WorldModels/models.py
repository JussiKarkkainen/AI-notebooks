import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
import haiku as hk
import optax
from typing import NamedTuple, Optional

class ConvVaeEncoder(hk.Module):
    padding = 'VALID' # Means no padding
    z_size = 32
    key = jax.random.PRNGKey(seed=42)

    def __call__(self, x):
        conv1 = hk.Conv2D(32, kernel_shape=4, stride=2, padding=self.padding)(x)
        conv1 = jax.nn.relu(conv1)
        conv2 = hk.Conv2D(64, kernel_shape=4, stride=2, padding=self.padding)(conv1)
        conv2 = jax.nn.relu(conv2)
        conv3 = hk.Conv2D(128, kernel_shape=4, stride=2, padding=self.padding)(conv2)
        conv3 = jax.nn.relu(conv3)
        conv4 = hk.Conv2D(256, kernel_shape=4, stride=2, padding=self.padding)(conv3)
        conv4 = jax.nn.relu(conv4)
        # Flattens on everything except batch dimension
        fc_in = hk.Flatten(preserve_dims=1)(conv4)
        # Is this correct ?
        mu = hk.Linear(self.z_size)(fc_in)
        logsigma = hk.Linear(self.z_size)(fc_in)
        return mu, logsigma
    
class ConvVaeDecoder(hk.Module):
    padding = 'VALID'
    fc_size = 1024

    def __call__(self, x):
        fc = hk.Linear(self.fc_size)(x)
        fc = fc.reshape(-1, 1, 1, 1024)
        conv1 = hk.Conv2DTranspose(128, kernel_shape=5, stride=2, padding=self.padding)(fc)
        conv1 = jax.nn.relu(conv1)
        conv2 = hk.Conv2DTranspose(64, kernel_shape=5, stride=2, padding=self.padding)(conv1)
        conv2 = jax.nn.relu(conv2)
        conv3 = hk.Conv2DTranspose(32, kernel_shape=6, stride=2, padding=self.padding)(conv2)
        conv3 = jax.nn.relu(conv3)
        conv4 = hk.Conv2DTranspose(3, kernel_shape=6, stride=2, padding=self.padding)(conv3)
        conv4 = jax.nn.sigmoid(conv4)
        return conv4

class ConvVAE(hk.Module):
    '''
    Vision model is a variational autoencoder
    '''
    key = jax.random.PRNGKey(seed=42)

    def __call__(self, x):
        # See figure 22 -> https://arxiv.org/pdf/1803.10122.pdf
        mu, logsigma = ConvVaeEncoder()(x)
        sigma = jnp.exp(logsigma)
        z = mu + sigma * jax.random.normal(self.key, mu.shape)
        decoded = ConvVaeDecoder()(z)
        return z, mu, logsigma, decoded

class MDNLSTMState(NamedTuple):
    hidden: jnp.ndarray
    cell: jnp.ndarray

class MDN_LSTM(hk.RNNCore):
    '''
    An LSTM core with MDN output
    '''
    def __init__(self, hidden_units=256, n_gaussian=5):
        super().__init__()
        self.hidden_units = hidden_units
        self.n_gaussian = n_gaussian

    # See figure 23 from paper
    def __call__(self, inputs, prev_state):
        # LSTM 
        z, a = inputs[0], inputs[1]
        z_a = jnp.concatenate([a, z], axis=-1)
        z_h = jnp.concatenate([z_a, prev_state.hidden], axis=-1)
        # 4x here because this will be split into 4 parts
        gated = hk.Linear(4*self.hidden_units)(z_h)
        # i = input, g = cell_gate, f = forget_gate, o = output_gate
        i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
        f = jax.nn.sigmoid(f + 1)
        c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        # MDN
        outs = hk.Linear(3*self.n_gaussian)(h)
        alpha, mu, logsigma = jnp.split(outs, indices_or_sections=3, axis=-1)
        logsigma -= logsumexp(logsigma, axis=-1, keepdims=True)
        return (h, alpha, mu, logsigma), MDNLSTMState(hidden=h, cell=c)
    
    def initial_state(self, batch_size):
        state = MDNLSTMState(hidden=jnp.zeros([self.hidden_units]),
                          cell=jnp.zeros([self.hidden_units]))
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state

def add_batch(nest, batch_size: Optional[int]):
    """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
    broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
    return jax.tree_util.tree_map(broadcast, nest)

def mixture_coef(n_gaussian, h):
    alpha = hk.Linear(self.n_gaussian)(h)
    mu = hk.Linear(self.n_gaussian)(h) 
    logsigma = hk.Linear(n_gaussian)(h)
    sigma = jnp.exp(logsigma)
    alpha = jax.nn.softmax(pi)
    return alpha, mu, sigma

class MDM_RNN(hk.Module):
    def __init__(self):
        super().__init__()
        self.n_gaussian = 5
        self.hidden_units = 256

    def __call__(self, z, a, prev_state):
        # Takes hidden state of LSTM as input and produces
        # a propability distribution over z
        h, (c, o) = LSTM()(z, a, prev_state)
        pi, mu, sigma = self.mixture_coef(h)
        return (pi, mu, sigma), (h, c)
    
    


class Controller(hk.Module):
    fc_size = 3

    def __call__(self, z, h):
        # Linear layer that maps the concatenated input vector [z, h]
        # into an action vector
        z_h = jnp.concatenate(z, h, axis=1)
        return hk.Linear(fc_size)(z_h)

