#from google.colab import files
import pickle
import sklearn
from sklearn.metrics import classification_report
"""
def upload():
  data = files.upload()
  return data
"""
def read_youtube(): # returns texts and the assoicated labels seperatly
  with open ('/content/drive/MyDrive/bert_deepmoji/data/SS-Youtube/raw.pickle','rb') as dataset:
    data = pickle.load(dataset)

  try:
      texts = [x for x in data['texts']]
  except UnicodeDecodeError:
      texts = [x.decode('utf-8') for x in data['texts']]

    # Extract labels
  labels = [x['label'] for x in data['info']]

  return texts, labels
  
def read_twitter(): # returns texts and the assoicated labels seperatly
  with open ('/content/drive/MyDrive/bert_deepmoji/data/SS-Twitter/raw.pickle','rb') as dataset:
    data = pickle.load(dataset)

  try:
      texts = [x for x in data['texts']]
  except UnicodeDecodeError:
      texts = [x.decode('utf-8') for x in data['texts']]

    # Extract labels
  labels = [x['label'] for x in data['info']]

  return texts, labels

def encode_output(results): # encodes the output so we can use in confusion matrix
  for i in range(len(results)):
    if results[i]['label'] == 'POSITIVE':
      results[i]['label'] = 1
    else:
      results[i]['label'] = 0
  return results

def print_metrics(results, labels): # outputs confusion matrix
  tmp = []
  for i in range(len(results)):
    tmp.append(results[i]['label'])
  print_metrics(classification_report(labels,tmp))
