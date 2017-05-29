def load_vocab(filename):
  lines = open(filename).readlines()
  return {
    word.strip() : i
    for i,word in enumerate(lines)
  }

vocab = load_vocab('data/vocabulary.txt')

def process_train(row):
  context,response,label = row

  context = list(map(lambda k: vocab.get(k, 0), context))
  response = list(map(lambda k: vocab.get(k, 0), response))
  label = int(label)

  return context,response,label
