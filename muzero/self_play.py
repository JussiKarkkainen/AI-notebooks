from games import MuZeroConfig
import games
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
from network import MuZeroNetwork

class SelfPlay:
    '''
    Gather data to replay buffer and play game during test time
    '''

    def __init__(self, game, config: MuZeroConfig, init=False):
        self.config = config
        self.init = init
        self.model = MuZeroNetwork(config, init)
        self.game = games.Game() 
        

    # Calls play_game() and stores game data to replay buffer and networks to shared storage
    def run_selfplay(self, shared_storage: SharedStorage, replay_buffer: ReplayBuffer):
        while shared_storage.training_step < self.config.training_steps:
            # TODO Need to add seperate cases for training and inference
            game_history = self.play_game(self.config, self.model)
            replay_buffer.save_game(game_history)
            shared_storage.increment_trainingstep()            

    # Execute Monte Carlo Tree Search to generate moves 
    def play_game(self, config: MuZeroConfig, network: MuZeroNetwork):
        game_history = GameHistory()



class GameHistory:

    def __init__(self):
        pass


class MCTS:
    def __init__(self):
        pass
