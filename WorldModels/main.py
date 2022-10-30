import sys
import weights
from dataset import Dataset
import trainer

class WorldModel:
    def __init__(self):
        pass

    def train(self):
        print("Creating dataset")
        print("___________________\n\n")
        dataset, episodes = Dataset().rollout()
        print("Finished creating dataset for VAE\n")
        print("Starting VAE training\n")
        vae_model_state, model, vae_encode_batch = trainer.VAETrainer(dataset, episodes).train()
        print("Finished training VAE\n")
        weights.save_model(vae_model_state, name="vae")
        print("Starting LSTM training\n")
        lstm_model_state = trainer.LSTMTrainer(dataset, vae_encode_batch, episodes, 
                                               vae_model_state.params, model).train()
        weights.save_model(lstm_model_state, name="lstm")

    def test(self, path=None):
        print("Starting test")
        print("________________\n\n")
        model = weigths.load_model("VAE")
        Test(self.config).test()


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise Exception("Invalid amount of arguments, use either '--train' or '--test'")
    if sys.argv[1] == "--train":
        WorldModel().train()
    elif sys.argv[1] == "--test":
        WorldModel().test()

