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
    
    c = Variable(torch.from_numpy(np.array(context)), volatile=True).cuda()
    r = Variable(torch.from_numpy(np.array(response)), volatile=True).cuda()
    
    results = [model(c, r)]
    for dist in distractors:
      d = Variable(torch.from_numpy(np.array(dist)), volatile=True).cuda()
      results.append(model(c, d))
      
    results = [e.data.cpu().numpy()[0,0] for e in results]

    better_count = sum(1 for val in results[1:] if val >= results[0])
    count[better_count] += 1

  return count
