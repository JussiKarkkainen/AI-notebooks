import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import skimage

class ReplayBuffer:
    def __init__(self, seq_len):
        self.buffer = []
        self.act_buffer = []
        self.reward_buffer = []
        self.seq_len = seq_len

    def save(self, observation, action, reward):
        self.buffer.append(jnp.array(observation))
        self.act_buffer.append(jnp.array(action))
        self.reward_buffer.append(jnp.array(reward))

    def get_image(self):
        return self.preprocess(self.buffer[100])

    def seq_getter(self, batch_size, targets=False):
        buf = self.buffer[1:] if targets else self.buffer[:len(self.buffer)-1]
        dataset = []
        index = 0 
        for k in range(len(buf) // (batch_size*self.seq_len)):
            batches = []
            for j in range(batch_size):
                batch = []
                for i in range(self.seq_len):
                    batch.append(self.preprocess(buf[i+index]))
                batches.append(batch)
                index += self.seq_len
            dataset.append(batches)
        dataset = jnp.array(dataset)
        print(f"Shape of MDN_LSTM (targets: {targets}) dataset is: {dataset.shape}")
        return dataset
    
    def get_train_actions(self, batch_size):
        buf = self.act_buffer[:len(self.act_buffer)-1]
        dataset = []
        index = 0
        for k in range(len(buf) // (batch_size*self.seq_len)):
            batches = []
            for j in range(batch_size):
                batch = []
                for i in range(self.seq_len):
                    batch.append(buf[i+index])
                batches.append(batch)
                index += self.seq_len
            dataset.append(batches)
        dataset = jnp.array(dataset)
        print(f"Shape of actions dataset is: {dataset.shape}")
        return dataset 
    
    def get_latents(self, obs, targets, v_model, v_model_params):
        ''' 
        Turn observations to latent vectors so they can be used by M-model
        obs.shape is (n, bs, seq_len, h, w, c) loop over batches and sequences to produce latents
        '''
        # TODO: check if correct, also very slow
        latents, target_latents = [], []
        for k in range(obs.shape[0]):
            o_n, t_n = [], []
            for i in range(obs.shape[1]):
                o_batch, t_batch = [], []
                for j in range(obs.shape[2]):
                    v_input = jnp.expand_dims(obs[k, i, j, :, :, :], axis=0)
                    t_input = jnp.expand_dims(targets[k, i, j, :, :, :], axis=0)
                    o_batch.append(v_model.apply(v_model_params, v_input)[0])
                    t_batch.append(v_model.apply(v_model_params, t_input)[0])
                o_n.append(jnp.array(o_batch))
                t_n.append(jnp.array(t_batch))
            latents.append(jnp.array(o_n))
            target_latents.append(jnp.array(t_n))
        latents = jnp.squeeze(jnp.array(latents))
        target_latents = jnp.squeeze(jnp.array(target_latents))
        print(f"Shape of latents dataset is: {latents.shape}")
        print(f"Shape of target_latents dataset is: {target_latents.shape}")
        return latents, target_latents

    def get_train_inputs(self, batch_size):
        dataset = []
        index = 0
        for i in range(0, (len(self.buffer) // batch_size)): 
            batch = []
            for j in range(batch_size):
                batch.append(self.preprocess(self.buffer[j+index]))
            index += batch_size
            dataset.append(batch)
        dataset = jnp.array(dataset)
        print(f"Shape of VAE dataset is: {dataset.shape}")
        return dataset 
   
    def preprocess(self, image):
        image = jnp.array(image)
        image /= 255
        image = jnp.array(skimage.transform.resize(image, (64, 64)))
        return image
