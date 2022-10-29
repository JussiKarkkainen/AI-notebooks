import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple
from models import ConvVAE

class VAETrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset 
        self.forward = hk.without_apply_rng(hk.transform(self._forward))
        self.rng = jax.random.PRNGKey(seed=42)
        self.batch_size = 32

    def _forward(self, x):
        net = ConvVAE()
        z, decoded = net(x)
        return decoded

    @jax.jit
    def loss_fn(self):
        pass

    @jax.jit
    def update_weigths(self):
        pass

    def optimizer(self, lr):
        return optax.adam(lr)

    def make_initial_state(self, rng, x):
        init_params = self.forward.init(rng, x)
        opt_state = self.optimizer(0).init(init_params)
        return VAETrainingState(params=init_params, opt_state=opt_state)

    def train(self):
        initial_state = self.make_initial_state(self.rng, self.dataset.get_batch(self.batch_size))
         


