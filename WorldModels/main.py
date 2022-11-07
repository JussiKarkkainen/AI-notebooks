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
        #self.vae_model_state, self.model, vae_encode_batch = trainer.VTrainer(self.dataset, 
        #                                                                        episodes).train()
        print("Finished training VAE\n")
        #weights.save_model(self.vae_model_state, name="vae")
        vae_params = weights.load_model("vae")
        print("Starting LSTM training\n")
        mdn_rnn_params = trainer.MTrainer(self.dataset, episodes, vae_params).train()
        #weights.save_model(lstm_model_state, name="mdn_rnn")
        #controller_params = trainer.CTrainer(self.dataset, episodes, mdn_rnn_params).train()
        #self.test()


    def test(self, path=None):
        print("Starting test")
        print("________________\n\n")
        model_params = weights.load_model("vae")
        Test(model_params, self.dataset).test_vae()
        reward = Test(v_params, v_model, m_params, m_model, c_params, c_model).unroll()
        print(f"Reward on CarRacing-v2 was {reward}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise Exception("Invalid amount of arguments, use either '--train' or '--test'")
    if sys.argv[1] == "--train":
        WorldModel().train()
    elif sys.argv[1] == "--test":
        WorldModel().test()

