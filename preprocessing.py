def load_vocab(filename):
  lines = open(filename).readlines()
  return {
    word.strip() : i
    for i,word in enumerate(lines)
  }

vocab = load_vocab('data/vocabulary.txt')

def load_glove_embeddings(filename='data/glove.6B.100d.txt'):
  lines = open(filename).readlines()
  embeddings = {}
  for line in lines:
    word = line.split()[0]
    embedding = list(map(float, line.split()[1:]))
    if word in vocab:
      embeddings[vocab[word]] = embedding
  
  return embeddings
  
def numberize(inp):
  result = list(map(lambda k: vocab.get(k, 0), inp))[-160:]
  if len(result) < 160:
    result += [0]*(160 - len(result))

  return result

def process_train(row):
  context,response,label = row

  context = numberize(context)
  response = numberize(response)
  label = int(label)

  return context,response,label

def process_valid(row):
  context = row[0]
  response = row[1]
  distractors = row[2:]

  context = numberize(context)
  response = numberize(response)
  distractors = [
    numberize(distractor)
    for distractor in distractors
  ]

  return context, response, distractors
