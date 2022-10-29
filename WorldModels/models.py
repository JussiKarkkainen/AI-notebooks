import jax.numpy as jnp
import jax
import haiku as hk
import optax
from typing import NamedTuple, Optional

class ConvVaeEncoder(hk.Module):
    stride = 2
    padding = 'VALID' # Means no padding
    kernel_size = 4
    out_channels_1 = 32
    out_channels_2 = 64
    out_channels_3 = 128
    out_channels_4 = 256
    fc_size = 32
    key = jax.random.PRNGKey(seed=42)

    def __call__(self, x):
        # Input is 64x64x3
        # 4 conv layers
        # 1. relu conv, out_channels=32, kernel_size=4, stride=2
        # 2. relu conv, out_channels=64, kernel_size=4, stride=2
        # 3. relu conv, out_channels=128, kernel_size=4, stride=2
        # 4. relu conv, out_channels=256, kernel_size=4, stride=2
        # Latent vector Z_n = 32
        conv1 = hk.Conv2D(self.out_channels_1, self.kernel_size, stride=self.stride, padding=self.padding)(x)
        conv1 = jax.nn.relu(conv1)
        conv2 = hk.Conv2D(self.out_channels_2, self.kernel_size, stride=self.stride, padding=self.padding)(conv1)
        conv2 = jax.nn.relu(conv2)
        conv3 = hk.Conv2D(self.out_channels_3, self.kernel_size, stride=self.stride, padding=self.padding)(conv2)
        conv3 = jax.nn.relu(conv3)
        conv4 = hk.Conv2D(self.out_channels_4, self.kernel_size, stride=self.stride, padding=self.padding)(conv3)
        conv4 = jax.nn.relu(conv4)
        # Flattens on everything except batch dimension
        fc_in = hk.Flatten(preserve_dims=1)(conv4)
        # Is this correct ?
        mu = hk.Linear(self.fc_size)(fc_in)
        sigma = hk.Linear(self.fc_size)(fc_in)
        z = self.reparameterize(mu, sigma)
        return z
    
    # Check if this is correct
    def reparameterize(self, mu, sigma):
        std = jnp.exp(0.5*sigma)
        eps = jax.random.normal(self.key, std.shape)
        return jnp.add(jnp.multiply(eps, std), mu)

class ConvVaeDecoder(hk.Module):
    stride = 2
    padding = 'VALID'
    kernel_size = 5
    out_channels_1 = 128
    out_channels_2 = 64
    out_channels_3 = 32
    out_channels_4 = 3
    

    def __call__(self, x):
        # (H, W, C)
        # Input is 32
        # Input is 1x1x1024
        # 4 conv layers
        # 1. relu deconv, out_channels=128, kernel_size=5, stride=2
        # 2. relu deconv, out_channels=64, kernel_size=5, stride=2
        # 3. relu deconv, out_channels=32, kernel_size=6, stride=2
        # 4. sigmoid deconv, out_channels=3, kernel_size=6, stride=2
        fc = hk.Linear(1024)(x)
        fc = fc.reshape(-1, 1, 1, 1024)
        conv1 = hk.Conv2DTranspose(self.out_channels_1, self.kernel_size, stride=self.stride, padding=self.padding)(fc)
        conv1 = jax.nn.relu(conv1)
        conv2 = hk.Conv2DTranspose(self.out_channels_2, self.kernel_size, stride=self.stride, padding=self.padding)(conv1)
        conv2 = jax.nn.relu(conv2)
        conv3 = hk.Conv2DTranspose(self.out_channels_3, self.kernel_size, stride=self.stride, padding=self.padding)(conv2)
        conv3 = jax.nn.relu(conv3)
        conv4 = hk.Conv2DTranspose(self.out_channels_4, self.kernel_size, stride=self.stride, padding=self.padding)(conv3)
        conv4 = jax.nn.sigmoid(conv4)
        print(conv4.shape)
        return conv4

class ConvVAE(hk.Module):
    '''
    Vision model is a variational autoencoder
    '''
    def __call__(self, x):
        # See figure 22 -> https://arxiv.org/pdf/1803.10122.pdf
        encoded = ConvVaeEncoder()(x)
        decoded = ConvVaeDecoder()(encoded)
        return encoded, decoded

class LSTMstate(NamedTuple):
    hidden: jnp.ndarray
    cell: jnp.ndarray


class LSTM(hk.Module):
    hidden_units = 256

    # See figure 23 from paper
    def __call__(self, x, action, hidden):
        # how does action get included, maybe concat with hidden?
        x_h = jnp.concatenate([x, hidden], axis=-1)
        # 4x here because this will be split into 4 parts
        gated = hk.Linear(4*hidden_units)(x_h)
        # i = input, g = cell_gate, f = forget_gate, o = output_gate
        i, g, f, o = jnp.split(gated, indices_or_section=4, axis=-1)
        f = jax.nn.sigmoid(f + 1)
        c = f * hidden.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, LSTMstate(c, o)
        
    
    def initial_state(self, batch_size):
        state = LSTMstate(hidden=jnp.zeros([self.hidden_units]),
                          cell=jnp.zeros([self.hidden_size]))
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state

def add_batch(nest, batch_size: Optional[int]):
    """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
    broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
    return jax.tree_util.tree_map(broadcast, nest)


class Controller(hk.Module):
    # What size?
    fc_size = 128

    def __call__(self, z, h):
        # Linear layer that maps the concatenated input vector [z, h]
        # into an action vector
        z_h = jnp.concatenate(z, h, axis=1)
        a_t = hk.Linear(fc_size)(z_h)
        return a_t

class FullModel(hk.Module):
    batch_size = 32
    init = True
    #TODO: Figure out action space
    action_space = 3

    def __call__(self, x):
        # Take raw observation from env and feed it into the V model (VAE) to
        # produce latent vector z_t. z_t is fed into the C model (1 layer mlp)
        # and concatenated with M's (MDN_RNN) hidden state h_t. C outputs an 
        # action vector a_t. M will take z_t and a_t as input to update h_t
        # to produce h_t+1.
        z_t, decoded = ConvVAE()(x)
        if init:
            h_t = LSTM.initial_state(batch_size)
        #TODO: Check axis
        o_t = jnp.concatenate(z_t, h_t, axis=1)
        a_t = hk.Linear(action_space)(o_t)
        # TODO: Combine action and initial state ?
        h_t, state = LSTM()(z_t, action, initial_state)
