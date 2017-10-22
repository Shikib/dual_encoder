import nltk
from nltk.stem import SnowballStemmer

def load_vocab(filename):
  lines = open(filename).readlines()
  return {
    word.strip() : i
    for i,word in enumerate(lines)
  }

vocab = load_vocab('/home/ubuntu/dual_encoder/data/vocabulary.txt')

def load_glove_embeddings(filename='data/glove.6B.100d.txt'):
  lines = open(filename).readlines()
  embeddings = {}
  for line in lines:
    word = line.split()[0]
    embedding = list(map(float, line.split()[1:]))
    if word in vocab:
      embeddings[vocab[word]] = embedding
  
  return embeddings
  
def numberize(inp, pad_len=100):
  inp = inp.split()
  result = list(map(lambda k: vocab.get(k, 0), inp))[-pad_len:]
  if len(result) < pad_len:
    result = [0]*(pad_len - len(result)) + result

  return result

def process_train(row, pad_len=100):
  context,response,label = row

  context = numberize(context, pad_len=pad_len)
  response = numberize(response, pad_len=pad_len)
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

stemmer = SnowballStemmer("english")
def process_predict_embed(response):
  response = ' '.join(list(map(stemmer.stem, nltk.word_tokenize(response))))
  response = numberize(response) 
  return response
