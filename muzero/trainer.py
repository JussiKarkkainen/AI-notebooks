from games import MuZeroConfig
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
import optax

def loss_function(params, batch):
    pass

class Trainer:
    def __init__(self, config: MuZeroConfig): 
        self.config = config
        model = MuZeroNetwork(config, init=True)
        lr = config.lr_init
        optimizer = optax.Adam(lr) 
        # TODO
        self.weigth_decay = config.weigth_decay

    def train_network(self, shared_storage: SharedStorage, replay_buffer: ReplayBuffer):
        for i in range(self.config.training_steps):
            if i % config.checkpoint_interval == 0:
                shared_storage.save_network(i, self.model)
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            self.update_weights(batch)
        shared_storage.save_network(config.training_steps, network)

     
    def update_weights(self, batch):
        '''
        Update weigths for single batch of replay buffer
        '''
        loss = 0
        for image, action, target in batch:
            init_out = self.model.initial_inference(image)
            predictions = [(1., init_out.value, init_out.reward, init_out.policy_logits)]

            # Dynamics needs to be trained on sequence of actions
            for action in actions:
                rec_out = self.model.recurrent_inference(init_out.hidden_state, action)
                predictions.append((1.0 / len(actions), rec_out.value, rec_out.reward, 
                    rec_out.policy_logits))
                
            for pred, target in zip(predictions, targets):
                gradient_scale, value, reward, policy = pred
                target_value, target_reward, target_policy = target
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
