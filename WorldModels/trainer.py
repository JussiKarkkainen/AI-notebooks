import jax
import jax.numpy as jnp
import haiku as hk
import optax
import wandb
from typing import NamedTuple
import models

class VAETrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

class VAETrainer:
    def __init__(self, dataset, episodes):
        self.dataset = dataset 
        self.model = hk.without_apply_rng(hk.transform(self._forward))
        self.rng = jax.random.PRNGKey(seed=42)
        self.batch_size = 32
        self.train_inputs = dataset.get_train_inputs(self.batch_size)
        self.update_weights = jax.jit(self._update_weights)
        self.loss_fn = jax.jit(self._loss_fn)
        self.optimizer = optax.adam(1e-3)
        self.VAEState = VAETrainingState(params=None, opt_state=None) 
        self.num_epochs = (episodes // self.batch_size) - 1
        wandb.init(project="WorldModel VAE")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": self.num_epochs,
            "batch_size": 32
        }
    
    def _forward(self, x):
        net = models.ConvVAE()
        z, mu, std, decoded = net(x)
        return decoded, mu, std

    def _loss_fn(self, params, inputs):
        ''' 
        L2 distance between the input image and the reconstruction in addition to KL loss.
        '''
        y_hat, mu, std = self.model.apply(params, inputs)
        l2 = jnp.sum(optax.l2_loss(y_hat, inputs))
        kld = 0.5 * jnp.sum(1 + jnp.log(jnp.power(std, 2)) - jnp.power(mu, 2) - jnp.power(std, 2))
        return l2 + kld

    def _update_weights(self, state, inputs):
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=0)
        loss, grads = grad_fn(state.params, inputs)
        updates, opt_state = self.optimizer.update(state.params, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return VAETrainingState(params=params, opt_state=opt_state), loss

    def make_initial_state(self, rng, x):
        init_params = self.model.init(rng, x)
        opt_state = self.optimizer.init(init_params)
        return VAETrainingState(params=init_params, opt_state=opt_state)

    def step(self):
        inputs = next(self.train_inputs)
        self.VAEState, loss = self.update_weights(self.VAEState, inputs)
        return loss

    def train(self):
        dummy_inputs = next(self.train_inputs)
        self.VAEState = self.make_initial_state(self.rng, dummy_inputs) 
        for i in range(self.num_epochs):
            loss = self.step()
            wandb.log({"loss": loss}) 

        return self.VAEState, self.model

class LSTMTrainingState(NamedTuple):
    params: hk.Params
    opt_state = optax.OptState

class LSTMTrainer:
    def __init__(self, dataset, episodes, v_model_params, v_model):
        self.dataset = dataset
        self.episodes = episodes
        self.v_model_params = v_model_params # Train M-model together with fixed params of V-model
        self.v_model = v_model
        self.batch_size = 32
        self.train_inputs = dataset.get_train_inputs(self.batch_size)
        self.train_actions = dataset.get_train_actions(self.batch_size)
        self.rng = jax.random.PRNGKey(seed=42)
        self.lstm_model = hk.without_apply_rng(hk.transform(self._forward))
        self.loss_fn = jax.jit(self._loss_fn)
        self.update_weights = jax.jit(self._update_weights)
        self.optimizer = optax.adam(1e-3)
        wandb.init(project="WorldModel LSTM")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": self.num_epochs,
            "batch_size": 32
        }
        
    def _forward(self, z, h, action):
        lstm_net = models.LSTM()
        state = lstm_net(z, h, action)
        return state

    def _loss_fn(self, params, inputs):
        y_hat, mu, std = self.v_model.apply(self.v_model_params, inputs)

    def _update_weights(self, state, inputs):
        pass

    def make_initial_state(self, rng, x):
        lstm_inital_state = models.LSTM.initial_state(self.batch_size)
        init_params = self.lstm_model.init(rng, x)
        opt_state = self.optimizer.init(init_params)
        return LSTMTrainingState(params=init_params, opt_state=opt_state)

    def step(self):
        pass

    def train(self):
        dummy_inputs = next(self.train_inputs)
        dummy_actions = next(self.train_actions)
        self.LSTMState = self.make_initial_state(self.rng, dummy_inputs, dummy_actions)
        for i in range(self.num_epochs):
            loss = self.step()
            wanb.log(["loss": loss])

        return self.LSTMState, self.lstm_model

