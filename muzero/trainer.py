from games import MuZeroConfig
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self, config: MuZeroConfig): 
        self.config = config

    def train_network(self, shared_storage: SharedStorage, replay_buffer: ReplayBuffer):
        pass
