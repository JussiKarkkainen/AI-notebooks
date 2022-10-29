import pickle

def save_model(model_state):
    params = model_state.params
    with open(f"weights.pickle", "wb") as out_file:
        pickle.dump(params, out_file)

def load_model(path):
    with open(f"weights.pickle", "rb") as in_file:
        params = pickle.load(in_file)
    return params
