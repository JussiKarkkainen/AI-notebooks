import sys
import argparse
from muzero import MuZero

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuZero")
    parser.add_argument('--train', help='choose game to train on: Supported games include "cartpole-v1"')
    parser.add_argument('--test', help='choose game to test on: Supported games include "cartpole-v1"')
    args = parser.parse_args()
    
    
    if args.train:
        muzero = MuZero(args.train, training=True)
        muzero.execute()
    elif args.test:
        muzero = MuZero(args.test, training=False)
        muzero.execute()
    elif args.train and args.test:
        print("Invalid arguments, choose either train or test but not both")
    else:
        # Default is test
        muzero = MuZero(args.test, training=False)
        muzero.execute()

