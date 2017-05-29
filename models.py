import torch

from torch import nn
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

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
    self.input_size = input_size
    self.hidden_size = hidden_size // self.num_directions
    self.num_layers = num_layers
    self.rnn_type = rnn_type

    self.embedding = nn.Embedding(vocab_size, input_size, sparse=False)

    if rnn_type == 'gru':
      self.rnn = nn.GRU(
        input_size,
        hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
      )
    else:
      self.rnn = nn.LSTM(
        input_size,
        hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
      )

  def forward(self, inp, hidden):
    emb = self.embedding(inp).view(1, 1, self.input_size)
    output, hidden = self.rnn(emb, hidden)
    return output, hidden

class DualEncoder(nn.Module):
  def __init__(self, encoder):
    super(DualEncoder, self).__init__()
    self.encoder = encoder
    self.M = Variable(
      torch.randn(
        self.encoder.hidden_size, 
        self.encoder.hidden_size,
      ).type(dtype), 
      requires_grad=True,
    ).cuda()

  def forward(self, context, response):
    context_hiddens = []
    context_h = None
    for word in context:
      context_o, context_h = self.encoder(word, context_h)
      context_hiddens.append(context_h)

    response_hiddens = []
    response_h = None
    for word in response:
      response_o, response_h = self.encoder(word, response_h)
      response_hiddens.append(response_h)

    if self.encoder.rnn_type == 'lstm':
      context_h = context_h[0]
      response_h = response_h[0]

    context_h = context_h.view(1, self.encoder.hidden_size)
    response_h = response_h.view(self.encoder.hidden_size, 1)
    ans = torch.mm(torch.mm(context_h, self.M), response_h)

    return torch.sigmoid(ans)
