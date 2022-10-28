import argparse

class WorldModel:
    def __init__(self):
        self.config = make_carracing_config()

    def train(self):
        print("Starting training")
        print("___________________\n\n")

        model = Trainer(self.config).train()
        weights.save_model(model)
        print("Finished training")

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
