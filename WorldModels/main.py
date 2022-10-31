import sys
import weights
from dataset import Dataset
import trainer
from test import Test

class WorldModel:
    def __init__(self):
        pass

    def train(self):
        print("Creating dataset")
        print("___________________\n\n")
        self.dataset, episodes = Dataset().rollout()
        print("Finished creating dataset for VAE\n")
        print("Starting VAE training\n")
        self.vae_model_state, self.model, vae_encode_batch = trainer.VAETrainer(self.dataset, episodes).train()
        print("Finished training VAE\n")
        weights.save_model(self.vae_model_state, name="vae")
        print("Starting LSTM training\n")
        #lstm_model_state = trainer.LSTMTrainer(dataset, vae_encode_batch, episodes, 
        #                                       vae_model_state.params, model).train()
        #weights.save_model(lstm_model_state, name="lstm")
        self.test()


    def test(self, path=None):
        print("Starting test")
        print("________________\n\n")
        model_params = weights.load_model("vae")
        Test(self.vae_model_state, self.model, self.dataset).test_vae()


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise Exception("Invalid amount of arguments, use either '--train' or '--test'")
    if sys.argv[1] == "--train":
        WorldModel().train()
    elif sys.argv[1] == "--test":
        WorldModel().test()

