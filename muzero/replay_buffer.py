

# Used to store data from previous games
class ReplayBuffer:
    def __init__(self, config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, unroll_steps, td_steps):
        image_batch, action_batch, target_batch = [], [], []
        image_batch.append([self.buffer[i].observation_history for i in range(self.batch_size)])
        action_batch.append([self.buffer[i].action_history for i in range(self.batch_size)])
        target_batch.append([self.buffer[i].reward_history for i in range(self.batch_size]])

        return (image_batch, action_batch, target_batch)
