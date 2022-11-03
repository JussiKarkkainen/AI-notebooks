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

class MDNRNNTrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

class MTrainer:
    def __init__(self, dataset, episodes, v_model_params):
        self.dataset = dataset
        
        self.episodes = episodes
        self.batch_size = 32
        self.train_inputs = dataset.seq_getter(self.batch_size)
        self.train_targets = dataset.seq_getter(self.batch_size, targets=True)
        self.train_actions = dataset.get_train_actions(self.batch_size)
        self.rng = jax.random.PRNGKey(seed=42)
        self.m_model = hk.without_apply_rng(hk.transform(self._unroll_rnn))
        self.loss_fn = self._loss_fn
        self.update_weights = self._update_weights
        self.optimizer = optax.adam(1e-4)
        self.num_epochs = (episodes // self.batch_size) - 1
        self.MDNRNNState = MDNRNNTrainingState(params=None, opt_state=None)
        
        self.v_model_params = v_model_params # Train M-model together with fixed params of V-model
        self.v_model = hk.without_apply_rng(hk.transform(self.v_forward)) 

        '''
        wandb.init(project="WorldModel MDNRNN")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": self.num_epochs,
            "batch_size": 32
        }
        '''
    def v_forward(self, x):
        v_model = models.ConvVAE()
        z, mu, logsigma, decoded = v_model(x)
        return z
    
    def _unroll_rnn(self, z, a):
        batch_size = z.shape[0]
        core = models.MDN_LSTM()
        initial_state = core.initial_state(batch_size)
        inputs = (z, a)
        h, mdnlstm_state = hk.dynamic_unroll(core, inputs, initial_state, time_major=False)  
        return mdnlstm_state

    #@partial(jax.jit, static_argnums=(0,))
    def _loss_fn(self, params, z, a, y):
        # MDM-RNN predicts z_t+1 = y
        (pi, mu, sigma), (c, h) = self.m_model.apply(params, z, a)
        dist = mu + sigma * jax.random.normal(jax.random.PRNGKey(42), mu.shape)
        loss = jnp.exp(jnp.log(y))
        loss = jnp.sum(loss * pi, axis=2)
        loss = -jnp.log(loss)
        return jnp.mean(loss)

    #@partial(jax.jit, static_argnums=(0,))
    def _update_weights(self, state, inputs, targets, actions):
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=0)
        loss, grads = grad_fn(state.params, inputs, actions, targets)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return MDNRNNTrainingState(params=params, opt_state=opt_state)

    def make_initial_state(self, rng, z, action):
        init_params = self.m_model.init(rng, z, action)
        opt_state = self.optimizer.init(init_params)
        return MDNRNNTrainingState(params=init_params, opt_state=opt_state)
    
    def get_latents(self, obs, targets):
        ''' 
        Turn observations to latent vectors so they can be used by M-model
        obs.shape is (bs, seq_len, h, w, c) loop over batches and sequences to produce latents
        '''
        # TODO: check if correct
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

    def step(self, z_t, z_t1, a):
        # z_t should be of shape (batch_size, seq_len, H, W, C) 
        self.MDNRNNState, loss = self.update_weights(self.MDNRNNState, z_t, z_t1, a)
        return loss

    def train(self):
        actions = self.train_actions
        train_inputs = jnp.expand_dims(self.train_inputs, axis=0)
        train_targets = jnp.expand_dims(self.train_targets, axis=0)
        latents, target_latents = self.get_latents(self.train_inputs, self.train_targets)
        latents, target_latents = jnp.squeeze(latents), jnp.squeeze(target_latents)
        self.MDNRNNState = self.make_initial_state(self.rng, latents, actions)
        # Latents should be of shape (n, bs, seq_len, h, w, c)
        for z_t, z_t1, a in zip(latents, target_latents, actions):
            loss = self.step(jnp.squeeze(z_t), jnp.squeeze(z_t1), a)
            #wandb.log(["loss": loss])

        return self.MDNRNNState, self.lstm_model

class CTrainer:
    def __init__(self, dataset, episodes, mparams):
        self.dataset = dataset
        self.episodes = epsiodes
        self.mparams = mparams
