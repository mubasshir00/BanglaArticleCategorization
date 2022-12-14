# -*- coding: utf-8 -*-
"""bengali_document_categorization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/eftekhar-hossain/Bengali-Document-Categorization/blob/master/bengali_document_categorization.ipynb
"""



"""#Libraries"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import seaborn as sns
import re,nltk,json, pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM,GRU
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
np.random.seed(42)
class color: # Text style
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
# Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# dataset path
dataset_path = './headlines.csv'
path = '/content/drive/My Drive/Colab Notebooks/NLP Projects/Document Categorization/'


import tensorflow as tf
print(tf.__version__)

print(tf.keras.__version__)

"""#Importing Dataset"""

# Read the data
data = pd.read_csv(dataset_path,encoding='utf-8')
print(f'Total number of Documents: {len(data)}')

data = data[['Text','Category']]

# Plot the Class distribution
sns.set(font_scale=1.4)
data['Category'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Number of Articles", labelpad=12)
plt.ylabel("Category", labelpad=12)
plt.yticks(rotation = 45)
plt.title("Dataset Distribution", y=1.02);

"""The dataset has total 12 News Categories data and politics class has maximum number of articles.

#Data Preparation and Cleaning
"""

# Cleaning Data [Remove unncessary symbols]
def cleaning_documents(articles):
      '''
      This function will clean the news articles by removing punctuation marks and stopwords.

      Args:
      articles: a news text(str)

      returns:
      news: clean text (str)
      '''
      news = articles.replace('\n',' ')
      news = re.sub('[^\u0980-\u09FF]',' ',str(news)) #removing unnecessary punctuation
      # stopwords removal
      stp = open('./bangla_stopwords.txt','r',encoding='utf-8').read().split()
      result = news.split()
      news = [word.strip() for word in result if word not in stp ]
      news =" ".join(news)
      return news

# Apply the function into the dataframe
data['cleaned'] = data['Text'].apply(cleaning_documents)

# sample_data =[0]
# for i in sample_data:
#   print('Original : ',data.Text[i])
#   print('CLeaned : ',data.cleaned[i])
#   print('Category : ',data.Category[i])

#stop words
stp = open('./bangla_stopwords.txt','r',encoding='utf-8').read().split()

file = open('./bangla_stopwords.pkl','wb')
pickle.dump(stp, file)

stp = open('./bangla_stopwords.pkl','rb')
stp = pickle.load(stp)
len(stp)

# need to handle low lenght document

data['Length'] = data.cleaned.apply(lambda x :len(x.split()))

dataset = data

dataset.to_csv('cleaned_article.csv')

dataset = pd.read_csv('./cleaned_article.csv')
dataset.columns

def data_summary(dataset):
  documents = []
  words = []
  u_words = []
  class_label = [k for k,v in dataset.Category.value_counts().to_dict().items()]
  for label in class_label:
    word_list = [word.strip().lower() for t in list(dataset[dataset.Category == label].cleaned) for word in t.strip().split()]
    counts = dict()
    for word in word_list:
      counts[word] = counts.get(word,0)+1
    ordered = sorted(counts.items(),key= lambda item: item[1],reverse= True)
    documents.append(len(list(dataset[dataset.Category==label].cleaned)))
    words.append(len(word_list))
    u_words.append(len(np.unique(word_list)))
    
  return documents,words,u_words,class_label

documents,words,u_words,class_names = data_summary(dataset)

data_matrix = pd.DataFrame({'Total Docuements' : documents, 'Total Words' : words, 'Unique Words' : u_words,'Class Names': class_names})

# label encoding and dataset splitting

def label_encoding(dataset,bool):
  le = LabelEncoder()
  le.fit(dataset.Category)
  encoded_labels = le.transform(dataset.Category)
  labels = np.array(encoded_labels)
  class_names = le.classes_
  if bool == True:
    print(le.classes_)
  return labels

def dataset_split(news,category):
  X,X_test,y,y_test = train_test_split(news,category,train_size = 0.9,test_size = 0.1, random_state=0)

  X_train,X_valid,y_train,y_valid = train_test_split(X,y,train_size = 0.8,test_size = 0.2,random_state=0)

  return X_train,X_valid,X_test,y_train,y_valid,y_test


#Tokenizer

def encoded_texts(dataset,padding_length,max_words):
  tokenizer = Tokenizer(num_words = max_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n-',split=' ',char_level= False,oov_token='<oov>',document_count=0)

  tokenizer.fit_on_texts(dataset.cleaned)

  (word_counts,word_docs,word_index,document_count) = (tokenizer.word_counts,tokenizer.word_docs,tokenizer.word_index,tokenizer.document_count)

  def tokenizer_info(mylist,bool):
    ordered = sorted(mylist.items(),key= lambda item: item[1],reverse=bool)
    for w,c in ordered[:10]:
      print(w,"\t",c)
  tokenizer_info(word_counts, bool = True)
  tokenizer_info(word_docs, bool = True)
  tokenizer_info(word_index, bool = True)

  # convert string into list of integer indices

  sequences = tokenizer.texts_to_sequences(dataset.cleaned)
  word_index = tokenizer.word_index

  # print(dataset.cleaned[0],"\n",sequences[0])

  #pad sequences
  corpus = keras.preprocessing.sequence.pad_sequences(sequences,value=0.0,padding='post',maxlen= padding_length)
  # print(dataset.cleaned[0])
  # print(corpus[0])

  # label encoding
  labels = label_encoding(dataset, True)

  # save the tokenizer into a pickle file
  with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer, handle,protocol=pickle.HIGHEST_PROTOCOL)
  return corpus,labels

num_words = 5000
corpus,labels = encoded_texts(dataset, 300, num_words)

print(corpus.shape)


dataset_split(corpus, labels)