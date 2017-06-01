from torch import optim
from torch.autograd import Variable

import data
import datetime
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
batch_size = 100

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#print("Training started")
for i in range(num_epochs):

  #print("Starting new example", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  # Compute loss
  loss = 0

  batch = data.get_batch(i, batch_size)

  #print("Get batch", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  batch = list(map(preprocessing.process_train, batch))

  #print("Batch preprocessing done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  count = 0
  for c,r,y in batch:
    count += 1

    # Forward pass: compute predicted y by passing x to model
    c = torch.from_numpy(np.array(c))
    r = torch.from_numpy(np.array(r))
    y = torch.from_numpy(np.array([[y]])).float()

    y_pred = model(Variable(c).cuda(), Variable(r).cuda())

    # Compute and add loss
    if count != len(batch):
      loss += loss_fn(y_pred, Variable(y).cuda()).data[0]
    else:
      loss += loss_fn(y_pred, Variable(y).cuda())

    del y_pred, c, r, y

  #print("Batch forward done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  print(i, loss.data[0])

  # Before the backward pass, use the optimizer object to zero all of the
  # gradients for the variables it will update (which are the learnable weights
  # of the model)
  optimizer.zero_grad()

  #print("Zero grad done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  # Backward pass: compute gradient of the loss with respect to model parameters
  loss.backward()

  #print("Loss backward done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  # Calling the step function on an Optimizer makes an update to its parameters
  optimizer.step()

  #print("Optimizer step done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

  del loss, batch

import pdb; pdb.set_trace()
