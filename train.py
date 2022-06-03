#!/usr/bin/env python3

import sys
from MLP import *
from MLP import *

def call_nn(nn, dataset):
    if nn == "mlp" and dataset == "mnist":
        mlp_train_mnist()
    elif nn == "gru" and dataset == "ptb":
        gru_train_ptb()

if __name__ == "__main__":
    nn = sys.argv[1][2:]
    dataset = sys.argv[2][2:]

    call_nn(nn, dataset)

