from torch import optim
from torch.autograd import Variable

import data
import datetime
import evaluate
import models
import numpy as np
import preprocessing
import time
import torch

encoder_model = models.Encoder(
  input_size=100, # embedding dim 
  hidden_size=256, # rnn dim
  vocab_size=91620, # vocab size
  bidirectional=False, # really should change!
  rnn_type='lstm',
)
encoder_model.cuda()

model = models.DualEncoder(encoder_model)
model.cuda()

loss_fn = torch.nn.BCELoss()
loss_fn.cuda()

learning_rate = 0.001
num_epochs = 30000
batch_size = 128
evaluate_batch_size = 250

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#print("Training started")
for i in range(num_epochs):

  #print("Starting new example", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  batch = data.get_batch(i, batch_size)

  #print("Get batch", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  batch = list(map(preprocessing.process_train, batch))

  #print("Batch preprocessing done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  count = 0

  cs = []
  rs = []
  ys = []

  for c,r,y in batch:
    count += 1

    # Forward pass: compute predicted y by passing x to model
    cs.append(torch.LongTensor(c))
    #print("c", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    rs.append(torch.LongTensor(r))
    #print("r", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    ys.append(torch.FloatTensor([y]))
    #print("y", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

  cs = Variable(torch.stack(cs, 0)).cuda()
  rs = Variable(torch.stack(rs, 0)).cuda()
  ys = Variable(torch.stack(ys, 0)).cuda()

  y_preds = model(cs, rs)
  #print("y_preds", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
 
  # Compute loss
  loss = loss_fn(y_preds, ys)
  #print("loss", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))


  #print("Batch forward done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  if i % 100 == 1:
    print(y_preds,ys)
    print(i, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    print(i, loss.data[0])

  if i % 1000 == 0:
    res = evaluate.evaluate(model, size=evaluate_batch_size)
    print(i)
    print("1in10: %0.2f, 2 in 10: %0.2f, 5 in 10: %0.2f" % (
      res[0]/evaluate_batch_size,
      sum(res[:2])/evaluate_batch_size,
      sum(res[:5])/evaluate_batch_size,
    ))
    print(res)

  if i % 10000 == 0 and i > 0:
    res = evaluate.evaluate(model, size=2000)
     
    one_in = res[0]/2000
    two_in = sum(res[:2])/2000
    three_in = sum(res[:5])/2000

    print("!!!!!!!!!!")
    print("1in10: %0.2f, 2 in 10: %0.2f, 5 in 10: %0.2f" % (
      one_in,
      two_in,
      three_in,
    ))
    print(res)

    if one_in > 0.45:
      import pdb; pdb.set_trace()


  # Before the backward pass, use the optimizer object to zero all of the
  # gradients for the variables it will update (which are the learnable weights
  # of the model)
  optimizer.zero_grad()

  #print("Zero grad done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  # Backward pass: compute gradient of the loss with respect to model parameters
  loss.backward()

  #print("M grad",model.M.grad)
  #print("emb grad",model.encoder.embedding.weight.grad)
  #print("hh grad",model.encoder.rnn.weight_hh_l0.grad)
  #print("ih grad",model.encoder.rnn.weight_ih_l0.grad)

  #print("Loss backward done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  #torch.nn.utils.clip_grad_norm(model.parameters(), 10)
  #torch.nn.utils.clip_grad_norm([model.M], 10)

  # Calling the step function on an Optimizer makes an update to its parameters
  optimizer.step()

  #print("Optimizer step done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  del loss, batch

import pdb; pdb.set_trace()
