import csv

reader = csv.reader(open('data/train.csv'))
rows = list(reader)[1:]

def get_batch(epoch, batch_size):
  start = epoch * batch_size % len(rows)
  return rows[start:start+batch_size]
