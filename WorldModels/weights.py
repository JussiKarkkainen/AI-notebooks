import pickle

def save_model(model_state, name):
    params = model_state.params
    with open(f"{name}_weights.pickle", "wb") as out_file:
        pickle.dump(params, out_file)

def load_model(name):
    with open(f"{name}_weights.pickle", "rb") as in_file:
        params = pickle.load(in_file)
    return params
