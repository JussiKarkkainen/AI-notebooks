import torch
import numpy as np


with open('ptb.train.txt') as f:
  lines = f.readlines()

def get_tokens():
  tokens = [list(line) for line in lines]
  return tokens

token = get_tokens()


def flatten(tokens):
  return [items for i in tokens for items in i]

tokens = flatten(token)
print(len(tokens))


def unique_char(tokens):
  uniq_tokens = []
  for i in tokens:
    if i not in uniq_tokens:
      uniq_tokens.append(i)
  return uniq_tokens

vocab = {}
for e, char in enumerate(uniq_tokens):
  vocab[char] = e


wiki_numerical = [vocab[char] for char in tokens]

def one_hot_data(numerical_list, vocab_size=50):
    result = torch.zeros((len(numerical_list), vocab_size))
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

def textify(embedding):
    result = ""
    indices = torch.argmax(embedding, axis=1)
    for idx in indices:
        result += uniq_tokens[int(idx)]
    return result

seq_length = 64
num_samples = (len(wiki_numerical) - 1) // seq_length
dataset = one_hot_data(wiki_numerical[:num_samples * seq_length]).reshape(num_samples, seq_length, len(uniq_tokens))
dataset.shape


batch_size = 32
num_batches = len(dataset) // batch_size
train_iter = dataset[:num_batches * batch_size].reshape((batch_size, num_batches, seq_length, len(uniq_tokens)))
train_iter = train_iter.swapaxes(0, 1)
train_iter = train_iter.swapaxes(1, 2)
train_iter.shape


labels = one_hot_data(wiki_numerical[1:num_samples * seq_length + 1]).reshape(batch_size, num_batches, seq_length, len(uniq_tokens))
labels = labels.swapaxes(0, 1)
labels = labels.swapaxes(1, 2)
labels.shape

def init_hidden():
  return torch.zeros((1, 256))

W_xh = torch.normal(0, 0.01, (50, 256), requires_grad=True)
W_hh = torch.normal(0, 0.01, (256, 256), requires_grad=True)
b_h = torch.zeros((1, 256), requires_grad=True)

W_xr = torch.normal(0, 0.01, (50, 256), requires_grad=True)
W_hr = torch.normal(0, 0.01, (50, 256), requires_grad=True)
b_r = torch.zeros(256, requires_grad=True)

W_hr = torch.normal(0, 0.01, (256, 256), requires_grad=True)
W_hz = torch.normal(0, 0.01, (256, 256), requires_grad=True)
b_z = torch.zeros(256, requires_grad=True)

W_hq = torch.normal(0, 0.01, (256, 50), requires_grad=True)
b_q = torch.zeros(50, requires_grad=True)

params = [W_xh, W_hh, b_h, W_xr, W_hr, b_r, W_hr, W_hz, b_z, W_hq, b_q]
for param in params:
  param.requires_grad_(True)


def net(input, state):
  W_xh, W_hh, b_h, W_xr, W_hr, b_r, W_hr, W_hz, b_z, W_hq, b_q = params
  H_t = state
  outputs = []
  Sigmoid = torch.nn.Sigmoid()
  Tanh = torch.nn.Tanh()
  for x in input:
    R_t = Sigmoid((x @ W_xr) + (H_t @ W_hr) + b_r)
    Z_t = Sigmoid((x @ W_xr) + (H_t @ W_hz) + b_z)
    cand_hid = Tanh(x @ W_xh + (R_t * H_t) @ W_hh + b_h)
    H_t = Z_t * state + (1 - Z_t) * cand_hid
    outputs.append(softmax(H_t @ W_hq + b_q))

  return (outputs, H_t)

def crossentropy(y_hat, y):
  return -torch.mean(torch.sum(y * torch.log(y_hat)))

def average_ce_loss(outputs, labels):
  assert(len(labels == len(outputs)))
  total_loss = 0
  for (outputs, labels) in zip(outputs, labels):
    total_loss = total_loss + crossentropy(outputs, labels)
  return total_loss / len(outputs)

def fix_p(p):
    if p.sum() != 1.0:
        p = p*(1./p.sum())
    return p

def predict(prefix, num_chars):
  string = prefix
  sample_state = init_hidden()
  string_numerical = [vocab[char] for char in prefix]
  input = one_hot_data(string_numerical)
  
  for i in range(num_chars):
    outputs, sample_state = rnn(input, sample_state)
    choice = np.random.choice(50, p=fix_p(np.asarray(outputs[-1][0])))
    string += uniq_tokens[choice]
    input = one_hot_data([choice])
  return string

def grad_clipping(net, theta):
    """Clip the gradient."""
    params = net
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def softmax(X):
  lin = (X - torch.max(X).reshape((-1, 1)))
  X_exp = torch.exp(lin)
  partition = X_exp.sum(1, keepdim=True)
  return X_exp / partition

num_epochs = 500
criterion = average_ce_loss
params = params
lr = 0.01
optimizer = torch.optim.SGD(params, lr)
rnn = net

for epoch in range(num_epochs):
  state = init_hidden()
  for i in range(num_batches):
    input = train_iter[i]
    train_labels = labels[i]
    state = state.detach()
    optimizer.zero_grad()
    y_hat, state = rnn(input, state)
    l = criterion(y_hat, train_labels)
    l.sum().backward()
    grad_clipping(params, 1)

    optimizer.step()

  with torch.no_grad():
    l_loss = criterion(y_hat, train_labels)
    print(f'loss on epoch {epoch} was {l_loss}')
    print(predict('on the other hand', 512))


