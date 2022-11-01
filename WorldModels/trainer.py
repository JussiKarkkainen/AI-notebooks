import jax
import jax.numpy as jnp
import haiku as hk
import optax
import wandb
from typing import NamedTuple
from functools import partial
import models

class VAETrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

class VTrainer:
    def __init__(self, dataset, episodes):
        self.dataset = dataset 
        self.model = hk.without_apply_rng(hk.transform(self._forward))
        self.rng = jax.random.PRNGKey(seed=42)
        self.batch_size = 32
        self.train_inputs = dataset.get_train_inputs(self.batch_size)
        self.update_weights = self._update_weights
        self.loss_fn = self._loss_fn
        self.optimizer = optax.adam(1e-4)
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
        z, mu, sigma, decoded = net(x)
        return z, mu, sigma, decoded

    @partial(jax.jit, static_argnums=(0,))
    def _loss_fn(self, params, inputs):
        ''' 
        L2 distance between the input image and the reconstruction in addition to KL loss.
        '''
        z, mu, logsigma, decoded = self.model.apply(params, inputs)
        l2 = jnp.sum(optax.l2_loss(decoded, inputs))
        kld = -0.5 * jnp.sum(1 + 2*logsigma - jnp.power(mu, 2) - jnp.exp(2*logsigma))
        return l2 + kld, (z, mu, logsigma)
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_weights(self, state, inputs):
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)
        # ((value, aux), grads)
        loss_aux, grads = grad_fn(state.params, inputs)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return VAETrainingState(params=params, opt_state=opt_state), loss_aux[0], loss_aux[1]

    def make_initial_state(self, rng, x):
        init_params = self.model.init(rng, x)
        opt_state = self.optimizer.init(init_params)
        return VAETrainingState(params=init_params, opt_state=opt_state)

    def step(self, inputs):
        self.VAEState, loss, aux = self.update_weights(self.VAEState, inputs)
        return loss, aux

    def train(self):
        dummy_inputs = next(self.train_inputs)
        self.VAEState = self.make_initial_state(self.rng, dummy_inputs) 
        for data in self.train_inputs:
            loss, aux = self.step(data)
            self.encode_buffer.append(aux)
            print(loss)
            #wandb.log({"loss": loss}) 

        return self.VAEState, self.model, self.encode_buffer

class LSTMTrainingState(NamedTuple):
    #lstm_state: models.LSTMstate
    params: hk.Params
    opt_state: optax.OptState

class MTrainer:
    def __init__(self, dataset, episodes, v_model_params):
        self.dataset = dataset
        
        self.episodes = episodes
        self.batch_size = 32
        self.train_inputs = dataset.get_train_inputs(self.batch_size)
        self.train_targets = dataset.get_seq_targets(self.batch_size)
        self.train_actions = dataset.get_train_actions(self.batch_size)
        self.rng = jax.random.PRNGKey(seed=42)
        self.m_model = hk.without_apply_rng(hk.transform(self._forward))
        self.loss_fn = jax.jit(self._loss_fn)
        self.update_weights = jax.jit(self._update_weights)
        self.optimizer = optax.adam(1e-4)
        self.num_epochs = (episodes // self.batch_size) - 1
        self.LSTMState = LSTMTrainingState(params=None, opt_state=None)
        
        self.v_model_params = v_model_params # Train M-model together with fixed params of V-model
        self.v_model = hk.without_apply_rng(hk.transform(self.v_forward)) 

        '''
        wandb.init(project="WorldModel LSTM")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": self.num_epochs,
            "batch_size": 32
        }
        '''
    def v_forward(self, x):
        v_model = models.ConvVae()
        z, mu, logsigma, decoded = v_model(x)
        return z
    
    def _unroll_rnn(self, z, a):
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
        init_params = self.m_model.init(rng, z, action)
        opt_state = self.optimizer.init(init_params)
        return LSTMTrainingState(params=init_params, opt_state=opt_state)
    
    def get_latents(self, obs, targets):
        ''' 
        Turn observations to latents so they can be used by M-model
        Obs.shape is (bs, seq_len, h, w, c) loop over batches and sequences to produce latents
        '''
        latents, target_latents = [], []
        for i in range(obs.shape[0]):
            o_batch, t_batch = [], []
            for j in range(obs.shape[1]):
                o_batch.append(self.v_model.apply(self.v_model_params, jnp.expand_dims(obs[i][j], axis=0)))
                t_batch.append(self.v_model.apply(self.v_model_params, jnp.expand_dims(targets[i][j], axis=0)))
            latents.append(jnp.array(o_batch))
            target_latents.append(jnp.array(t_batch))
        latents = jnp.array(latents)
        target_latents = jnp.array(target_latents)
        return latents, target_latents

    def step(self, x, a):
        # TODO: Input should be of shape (batch_size, seq_len, H, W, C) 
        z, mu, logsigma, decoded = self.v_model.apply(self.v_model_params, x) 
        self.LSTMState, loss = self.update_weights(self.LSTMState, z, a)
        return loss

    def train(self):
        actions = self.train_actions
        latents, target_latents = self.get_latents(self.train_inputs, self.train_targets)
        self.LSTMState = self.make_initial_state(self.rng, dummy_z, dummy_a)
        for z_t, z_t1, a in zip(latents, self.target_latents, actions):
            loss = self.step(z_t, z_t1, a)
            #wandb.log(["loss": loss])

        return self.LSTMState, self.lstm_model

