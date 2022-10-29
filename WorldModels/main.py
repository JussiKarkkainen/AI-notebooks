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
        print("Starting training\n")
        model_state = Trainer(dataset, episodes).train()
        print("Finished training")
        weights.save_model(model_state)

    def test(self, path=None):
        print("Starting test")
        print("________________\n\n")
        model = weigths.load_model(path)
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
