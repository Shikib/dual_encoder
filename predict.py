import numpy as np
import preprocessing
import torch

from torch.autograd import Variable

model = torch.load("/home/ubuntu/dual_encoder/SAVED_MODEL")
model.cuda()

val_cache = {}
def predict_val(context, response):
  if (context,response) in val_cache:
    return val_cache[(context, response)]
  
  c_num = preprocessing.process_predict_embed(context)
  r_num = preprocessing.process_predict_embed(response)
  c = Variable(torch.LongTensor([c_num]), volatile=True).cuda()
  r = Variable(torch.LongTensor([r_num]), volatile=True).cuda()

  res = model(c, r, [c_num])[0].data.cpu().numpy()[0]
  val_cache[(context, response)] = res
  return val_cache[(context, response)]

cache = {}
def predict(response):
  if response in cache:
    return cache[response]
  list_response = preprocessing.process_predict_embed(response)
  inp = Variable(torch.LongTensor([list_response]), volatile=True).cuda()
  res = model.encoder(inp)
  cache[response] = res[1][0][0][0].data.cpu().numpy()

  return cache[response]
