#!/usr/bin/env python3

import sys
from MLP import *

def call_nn(nn, dataset):
    if nn == "mlp" and dataset == "mnist":
        mlp_train_mnist()


if __name__ == "__main__":
    nn = sys.argv[1][2:]
    dataset = sys.argv[2][2:]

    call_nn(nn, dataset)

