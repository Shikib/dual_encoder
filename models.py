import preprocessing 
import datetime
import time
import torch

from torch import nn
from torch.nn import init
from torch.autograd import Variable

#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

class Encoder(nn.Module):
  def __init__(
    self,
    input_size,
    hidden_size,
    vocab_size,
    num_layers=1,
    dropout=0,
    bidirectional=True,
    rnn_type='gru',
  ):
    super(Encoder, self).__init__()
    self.num_directions = 2 if bidirectional else 1
    self.vocab_size = vocab_size
    self.input_size = input_size
    self.hidden_size = hidden_size // self.num_directions
    self.num_layers = num_layers
    self.rnn_type = rnn_type

    self.embedding = nn.Embedding(vocab_size, input_size, sparse=False, padding_idx=0)

    if rnn_type == 'gru':
      self.rnn = nn.GRU(
        input_size,
        hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        batch_first=True,
      ).cuda()
    else:
      self.rnn = nn.LSTM(
        input_size,
        hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        batch_first=True,
      ).cuda()

    self.init_weights()

  def forward(self, inps):
    embs = self.embedding(inps)
    outputs, hiddens = self.rnn(embs)
    return outputs, hiddens

  def init_weights(self):
    init.orthogonal(self.rnn.weight_ih_l0)
    init.uniform(self.rnn.weight_hh_l0, a=-0.01, b=0.01)

    glove_embeddings = preprocessing.load_glove_embeddings()
    embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
    init.uniform(embedding_weights, a=-0.25, b=0.25)
    for k,v in glove_embeddings.items():
      embedding_weights[k] = torch.FloatTensor(v)
    embedding_weights[0] = torch.FloatTensor([0]*self.input_size)
    del self.embedding.weight
    self.embedding.weight = nn.Parameter(embedding_weights)

def detach_all(var):
  return [e.detach() for e in var]

class DualEncoder(nn.Module):
  def __init__(self, encoder):
    super(DualEncoder, self).__init__()
    self.encoder = encoder
    M = torch.FloatTensor(self.encoder.hidden_size, self.encoder.hidden_size)
    init.normal(M)
    self.M = Variable(
      M,
      requires_grad=True,
    ).cuda()

  def forward(self, contexts, responses):
    #print("start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    context_os, context_hs = self.encoder(contexts)
    #print("context", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    response_os, response_hs = self.encoder(responses)
    #print("response", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

    if self.encoder.rnn_type == 'lstm':
      context_hs = context_hs[0]
      response_hs = response_hs[0]

    results = []
    for i in range(len(context_hs[0])):
      context_h = context_hs[0][i].view(1, self.encoder.hidden_size)
      response_h = response_hs[0][i].view(self.encoder.hidden_size, 1)

      ans = torch.mm(torch.mm(context_h, self.M), response_h)[0][0]
      results.append(torch.sigmoid(ans))

    results = torch.stack(results)
    #print("multiplies", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

    return results
