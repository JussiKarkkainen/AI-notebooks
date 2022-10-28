import jax.numpy as jnp

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
        # Return batch_size amount of games from buffer
        games = [self.sample_game(i) for i in range(self.batch_size)]
        print(games[0])
        # Sample game positions from games
        game_pos = [(g, self.sample_position(g)) for g in games]
        image_batch = [self.buffer[i].observation_history[:unroll_steps] for i in range(self.batch_size)]
        action_batch = [self.buffer[i].action_history[:unroll_steps] for i in range(self.batch_size)]
        target_batch = [self.buffer[i].reward_history[:unroll_steps] for i in range(self.batch_size)]
        return jnp.array(image_batch), jnp.array(action_batch), jnp.array(target_batch)
    
    
    def sample_game(self, index):
        return self.buffer[index]

    def sample_position(self, game):
        pass

    def make_target(self, state_index, num_unroll_steps, td_steps, to_play):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            # Root values?
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index], self.child_visits[current_index]))
            else:
                targets.append((0, 0, []))
        return targets
