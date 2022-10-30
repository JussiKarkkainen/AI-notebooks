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
        self.encode_buffer = []

        '''
        wandb.init(project="WorldModel VAE")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": self.num_epochs,
            "batch_size": 32
        }
        '''
    def _forward(self, x):
        net = models.ConvVAE()
        z, mu, std, decoded = net(x)
        return decoded, mu, std, z

    def _loss_fn(self, params, inputs):
        ''' 
        L2 distance between the input image and the reconstruction in addition to KL loss.
        '''
        y_hat, mu, std, z = self.model.apply(params, inputs)
        l2 = jnp.sum(optax.l2_loss(y_hat, inputs))
        kld = 0.5 * jnp.sum(1 + jnp.log(jnp.power(std, 2)) - jnp.power(mu, 2) - jnp.power(std, 2))
        return l2 + kld, (z, mu, std)

    def _update_weights(self, state, inputs):
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)
        # ((value, aux), grads)
        loss_aux, grads = grad_fn(state.params, inputs)
        updates, opt_state = self.optimizer.update(state.params, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return VAETrainingState(params=params, opt_state=opt_state), loss_aux[0], loss_aux[1]

    def make_initial_state(self, rng, x):
        init_params = self.model.init(rng, x)
        opt_state = self.optimizer.init(init_params)
        return VAETrainingState(params=init_params, opt_state=opt_state)

    def step(self):
        inputs = next(self.train_inputs)
        self.VAEState, loss, aux = self.update_weights(self.VAEState, inputs)
        return loss, aux

    def train(self):
        dummy_inputs = next(self.train_inputs)
        self.VAEState = self.make_initial_state(self.rng, dummy_inputs) 
        for i in range(self.num_epochs - 1):
            loss, aux = self.step()
            self.encode_buffer.append(aux)
            #wandb.log({"loss": loss}) 

        return self.VAEState, self.model, self.encode_buffer

class LSTMTrainingState(NamedTuple):
    #lstm_state: models.LSTMstate
    params: hk.Params
    opt_state: optax.OptState

class LSTMTrainer:
    def __init__(self, dataset, vae_encode_batch, episodes, v_model_params, v_model):
        self.dataset = dataset
        # TODO: create latent vectors for MDM-RNN at runtime instead of here, this doesn't scale
        self.vae_encode_batch = iter(jnp.array([v_model.apply(v_model_params, 
            dataset.get_train_inputs(32, iterator=False)[i])[3] for i in range((episodes // 32) - 1)]))
    
        
        self.episodes = episodes
        self.v_model_params = v_model_params # Train M-model together with fixed params of V-model
        self.v_model = v_model
        self.batch_size = 32
        self.train_labels = dataset.get_targets(self.batch_size)
        self.train_inputs = dataset.get_train_inputs(self.batch_size)
        self.train_actions = dataset.get_train_actions(self.batch_size)
        self.rng = jax.random.PRNGKey(seed=42)
        self.lstm_model = hk.without_apply_rng(hk.transform(self._forward))
        self.loss_fn = jax.jit(self._loss_fn)
        self.update_weights = jax.jit(self._update_weights)
        self.optimizer = optax.adam(1e-3)
        self.num_epochs = (episodes // self.batch_size) - 1
        self.LSTMState = LSTMTrainingState(params=None, opt_state=None)
        '''
        wandb.init(project="WorldModel LSTM")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": self.num_epochs,
            "batch_size": 32
        }
        '''
    def _forward(self, z, a):
        lstm_net = models.LSTM()
        lstm_initial_state = lstm_net.initial_state(32)
        state = lstm_net(z, a, lstm_initial_state)
        return state

    def _loss_fn(self, params, z, a, y):
        # MDM-RNN predicts z_t+1 = y
        state = self.lstm_model.apply(params, z, a)
        loss = optax.cross_entropy_loss(state, y)
        return loss
        
    def _update_weights(self, state, inputs, actions, targets):
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=0)
        loss, grads = grad_fn(state.params, inputs, actions, targets)
        updates, opt_state = self.optimizer.update(state.params, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return LSTMTrainingState(params=params, opt_state=opt_state)

    def make_initial_state(self, rng, z, action):
        init_params = self.lstm_model.init(rng, z, action)
        opt_state = self.optimizer.init(init_params)
        return LSTMTrainingState(params=init_params, opt_state=opt_state)

    def step(self):
        z = next(self.vae_encode_batch)
        a = next(self.train_actions)
        y = next(self.train_labels)
        self.LSTMState, loss = self.update_weights(self.LSTMState, z, a, y)
        return loss

    def train(self):
        # TODO: get rid of magic values
        dummy_z = next(self.vae_encode_batch)
        dummy_a = next(self.train_actions)
        self.LSTMState = self.make_initial_state(self.rng, dummy_z, dummy_a)
        for i in range(self.num_epochs - 1):
            loss = self.step()
            #wandb.log(["loss": loss])

        return self.LSTMState, self.lstm_model

