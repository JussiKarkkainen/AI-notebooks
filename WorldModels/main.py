import os
import sys
import weights
from dataset import Dataset
import trainer
from test import Test

class WorldModel:
    def __init__(self, vae_batch_size=32, batch_size=8):
        self.vae_batch_size = vae_batch_size
        self.batch_size = batch_size
        self.dataset = Dataset()
        self.buf, self.episodes = self.dataset.rollout()

    def create_vae_dataset(self):
        print("Creating dataset for VAE")
        vae_dataset = self.buf.get_train_inputs(self.vae_batch_size)
        print("Done, Saving dataset") 
        path = self.dataset.save(vae_dataset, name="vae_dataset") 
        return path

    def create_mdnrnn_dataset(self, path):
        print("Creating dataset for MDN-LSTM")
        vae_params = weights.load_model("vae")
        mdn_lstm_inputs = self.buf.seq_getter(self.batch_size)  
        mdn_lstm_targets = self.buf.seq_getter(self.batch_size, targets=True)
        mdn_lstm_actions = self.buf.get_train_actions(self.batch_size)
        mdn_latents, mdn_latent_targets = self.buf.get_latents(mdn_lstm_inputs, mdn_lstm_targets, 
                                                          self.vae_forward, vae_params) 
        print("Done, Creating dataset for Controller")
        controller_dataset = None   # TODO 
        print("Done, Saving dataset")
        path = self.dataset.save((mdn_lstm_actions, mdn_latents, mdn_latent_targets), 
                                 name="mdnrnn")
        return path

    def create_c_dataset(self, path):
        pass

    def train_vae(self, path, force=False):
        print("Training VAE")
        vae_dataset = self.dataset.load(path) 
        vae_trainer = trainer.VTrainer(vae_dataset, self.episodes)
        self.vae_forward = vae_trainer.model
        if not os.path.exists("vae_weights.pickle") or force:
            vae_state = vae_trainer.train()
            weights.save_model(self.vae_state, name="vae")

    def train_mdnrnn(self, vae_path, mdnrnn_path, force=False):
        print("Training M-model")
        vae_params = weights.load_model("vae")
        vae_dataset = self.dataset.load(vae_path) 
        mdn_lstm_actions, mdn_latent, mdn_latent_targets = self.dataset.load(mdnrnn_path) 
        mdnrnn_trainer = trainer.MTrainer(vae_dataset, mdn_lstm_actions, mdn_latent, 
                                          mdn_latent_targets, self.episodes, vae_params)
        self.mdnrnn_forward = mdnrnn_trainer.m_model 
        if not os.path.exists("mdn_rnn_weights.pickle") or force:
            mdn_rnn_state = mdnrnn_trainer.train()
            weights.save_model(mdn_rnn_state, name="mdn_rnn")

    def train_controller(self, path):
        print("Training controller")
        controller_state = trainer.CTrainer(self.dataset, episodes, mdn_rnn_state).train()
        weights.save_model(controller_state, name="controller")

    def test(self, path=None):
        v_params = weights.load_model("vae")
        m_params = weights.load_model("mdnrnn")
        c_params = weights.load_model("controller")
        Test(model_params, self.dataset).test_vae()
        reward = Test(v_params, v_model, m_params, m_model, c_params, c_model).unroll()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise Exception("Invalid amount of arguments, use either '--train' or '--test'")
    if sys.argv[1] == "--train":
        main = WorldModel()
        vae_path = "data/vae_dataset.npz"
        mdnrnn_path = "data/mdnrnn.npz"
        if not os.path.exists(vae_path):
            vae_path = main.create_vae_dataset()
        main.train_vae(vae_path)
        if not os.path.exists(mdnrnn_path):
            mdnrnn_path = main.create_mdnrnn_dataset(vae_path)
        main.train_mdnrnn(vae_path, mdnrnn_path)
    elif sys.argv[1] == "--test":
        WorldModel().test()



