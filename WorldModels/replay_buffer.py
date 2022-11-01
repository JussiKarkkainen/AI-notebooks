import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import skimage

class ReplayBuffer:
    def __init__(self, seq_len):
        self.buffer = []
        self.act_buffer = []
        self.seq_len = seq_len

    def save(self, observation, action):
        self.buffer.append(jnp.array(observation))
        self.act_buffer.append(jnp.array(action))
    
    def get_image(self):
        return self.preprocess(self.buffer[100])

    def get_seq_targets(self, batch_size):
        # TODO: return batch_size amount of batches
        dataset = []
        index = self.seq_len
        for i in range(len(self.buffer) // self.seq_len - 1):
            batch = []
            for j in range(self.seq_len):
                batch.append(self.preprocess(self.buffer[index+j]))
            index += self.seq_len
            dataset.append(batch)
        dataset = jnp.array(dataset)
        print(dataset.shape)
        raise Exception("dwe")
        return dataset
    
    def get_seq_inputs(self, batch_size):
        '''
        Returns batches of sequences of images
        '''
        # TODO: return batch_size amount of batches
        dataset = []
        index = 0
        for i in range(len(self.buffer) // self.seq_len - 1):
            batch = []
            for j in range(self.seq_len):
                batch.append(self.preprocess(self.buffer[index+j]))
            index += self.seq_len
            dataset.append(batch)
        dataset = jnp.array(dataset)
        print(dataset.shape)
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
    
    def get_train_actions(self, batch_size, iterator=True):
        dataset = []
        index = 0
        for i in range((len(self.act_buffer) // batch_size) - 1):
            batch = []
            for j in range(batch_size):
                batch.append(self.act_buffer[index+j])
            index += batch_size
            dataset.append(batch)
        dataset = jnp.array(dataset)
        return iter(dataset) if iterator else dataset

    def preprocess(self, image):
        image = jnp.array(image)
        image /= 255
        image = jnp.array(skimage.transform.resize(image, (64, 64)))
        return image
