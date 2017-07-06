import data
import numpy as np
import preprocessing
import torch

from torch.autograd import Variable

def evaluate(model, size=None):
  """
  Evaluate the model on a subset of vallidation set.
  """
  valid = list(map(preprocessing.process_valid, data.get_validation(size)))

  count = [0]*10

  for e in valid:
    context, response, distractors = e
    
    cs = Variable(torch.stack([torch.LongTensor(context) for i in range(10)], 0), volatile=True).cuda()
    rs = [torch.LongTensor(response)]
    rs += [torch.LongTensor(distractor) for distractor in distractors]
    rs = Variable(torch.stack(rs, 0), volatile=True).cuda()
    
    results, responses = model(cs, rs, [context for i in range(10)])
    results = [e.data.cpu().numpy()[0] for e in results]

    better_count = sum(1 for val in results[1:] if val >= results[0])
    count[better_count] += 1

  return count
