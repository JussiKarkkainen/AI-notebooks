from network import MuZeroNetwork
from games import MuZeroConfig
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
import optax

def loss_function(params, batch):
    pass

class Trainer:
    def __init__(self, config: MuZeroConfig): 
        self.config = config
        # self.model = MuZeroNetwork(config)
        self.lr = config.lr_init
        self.optimizer = optax.adam(self.lr) 
        self.weight_decay = config.weight_decay

    def train_network(self, shared_storage: SharedStorage, replay_buffer: ReplayBuffer):
        self.model = shared_storage.latest_network()
        for i in range(self.config.training_steps):
            if i % self.config.checkpoint_interval == 0:
                shared_storage.save_network(i, self.model)
            image_batch, action_batch, target_batch = replay_buffer.sample_batch(
                    self.config.num_unroll_steps, self.config.td_steps)
            self.update_weights(image_batch, action_batch, target_batch)
        shared_storage.save_network(self.config.training_steps, self.model)

     
    def update_weights(self, image_batch, action_batch, target_batch):
        '''
        Update weigths for single batch of replay buffer
        
        image_batch.shape = (batch_size, num_unroll_steps, observation_shape)
        reward_batch.shape = (batch_size, num_unroll_steps)
        target_batch.shape = (batch_size, num_unroll_steps)

        images.shape = (num_unroll_steps, observation_shape)
        actions.shape = (num_unroll_steps)
        target.shape = (num_unroll_steps)
        '''
        
        loss = 0
        init = True
        

        for images, actions, targets in zip(image_batch, action_batch, target_batch):
            
            images = images.reshape(images.shape[0], images.shape[1])
            actions = actions.reshape(actions.shape[0], 1)
            targets = targets.reshape(targets.shape[0], 1)
            
            # Images -> for each observation in batch, unroll the position num_unroll_steps into the future
            #           using actions provided in batch. Initial inference is used for the first observation
            #           to predict the value, reward and policy and compare these to the target value, target 
            #           reward and target policy
            init_out = self.model.initial_inference(images)
            
            predictions = [(1., init_out.value, init_out.reward, init_out.policy_logits)]

            # For subsequent actions, we will use the recurrent_inference function to predict the value, 
            # reward and policy and compare to the target value, target reward and target policy
            for i, action in enumerate(actions):
                rec_out = self.model.recurrent_inference(init_out.hidden_state[i], action)
                predictions.append((1.0 / len(actions), rec_out.value, rec_out.reward, 
                    rec_out.policy_logits))
                
                # hidden_state = scale_grad(hidden_state, 0.5)    

            # Compare predictions to targets given in batch
            for pred, target in zip(predictions, targets):
                gradient_scale, value, reward, policy = pred
                target_value, target_reward, target_policy = target
                # MSE or CrossEntropy ?
                l = ((value - target_value)**2 + (reward - target_reward)**2 + \
                        optax.softmax_cross_entropy(policy, target_policy))
                
                loss += scale_grad(l, gradient_scale)
        
        # Regularization loss
        for weigths in self.model.get_weigths():
            loss += self.weigth_decay * optax.l2_loss(weigths)

        optax.apply_updates(self.model.get_weights(), updates)


        # TODO Return losses for logging
        return
    
def scale_grad(value, scale):
    return value * scale + value * (1. - scale)
