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
        return dataset 

    def get_train_inputs(self, batch_size, iterator=True):
        dataset = []
        for i in range(0, len(self.buffer), batch_size):
            batch = []
            for j in range(batch_size):
                batch.append(self.preprocess(self.buffer[j+i]))
            dataset.append(batch)
        dataset = jnp.array(dataset)
        return iter(dataset) if iterator else dataset
   
    def preprocess(self, image):
        image = jnp.array(image)
        image /= 255
        image = jnp.array(skimage.transform.resize(image, (64, 64)))
        return image
