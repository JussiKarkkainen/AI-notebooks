import network 

# Save and retrieve latest neural network from store
class SharedStorage:
    def __init__(self):
        self.networks = {}
        self.training_step = 0

    def increment_trainingstep(self):
        self.training_step += 1

    def latest_network(self):
        if self.networks:
            return self.networks[max(self.network.keys())]
        else:
            return network.make_uniform_network()
    
    def save_network(self, step, network):
        self.networks[step] = network
