import argparse
import weights
from dataset import Dataset
from trainer import Trainer

class WorldModel:
    def __init__(self):
        pass

    def train(self):
        print("Creating dataset")
        print("___________________\n\n")
        dataset, episodes = Dataset().rollout()
        print("Finished creating dataset for VAE\n")
        print("Starting VAE training\n")
        vae_model_state, model = VAETrainer(dataset, episodes).train()
        print("Finished training VAE\n")
        weights.save_model(vae_model_state, name="vae")
        print("Starting LSTM training\n")
        lstm_model_state = LSTMTrainer(dataset, episodes, vae_model_state.params, model).train()
        weights.save_model(lstm_model_state, name="lstm")

    def test(self, path=None):
        print("Starting test")
        print("________________\n\n")
        model = weigths.load_model("VAE")
        Test(self.config).test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WorldWodels")
    parser.add_argument('--train', help='Train model on CarRacing-v2')
    parser.add_argument('--test', help='Test with pretrained weights on CarRacin-v2')

    args = parser.parse_args()

    if args.train:
        WorldModel().train()
    elif args.test:
        WorldModel().test()
    elif args.train and args.test:
        print("Invalid arguments, choose either train or test but not both")
    else:
        WorldModel().test()
