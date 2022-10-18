from games import MuZeroConfig
import games
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
import network

class SelfPlay:
    '''
    Gather data to replay buffer and play game during test time
    '''

    def __init__(self, game, config: MuZeroConfig, init=False):
        self.config = config
        self.init = init
        self.model = network.MuZeroNetwork(config, init)
         




    def run_selfplay(self, shared_storage: SharedStorage, replay_buffer: ReplayBuffer):
        
        
        

        pass

