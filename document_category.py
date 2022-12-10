import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import nltk
import re
from sklearn.metrics import classification_report
from tensorflow.keras.layers import LSTM, GRU
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
np.random.seed(42)


class color:  # Text style
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
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset_path = './headlines.csv'
path = '/content/drive/My Drive/Colab Notebooks/NLP Projects/Document Categorization/'
data = pd.read_csv(dataset_path, encoding='utf-8')
data = data[['Text', 'Category']]

# cleaning data remove symbols


def data_cleaning(articles):
    articles_data_clean = articles.replace('\n', ' ')
    articles_data_clean = re.sub(
        '[^\u0980-\u09FF]', ' ', str(articles_data_clean))
    stop_words = open('./bangla_stopwords.txt', 'r',
                      encoding='utf-8').read().split()
    result = articles_data_clean.split()
    articles_data_clean = [word.strip()
                           for word in result if word not in stop_words]
    articles_data_clean = " ".join(articles_data_clean)
    return articles_data_clean


data['Cleaned_data'] = data['Text'].apply(data_cleaning)

# stop words pickle
stp = open('./bangla_stopwords.txt', 'r', encoding='utf-8').read().split()

file = open('./bangla_stopwords.pkl', 'wb')
pickle.dump(stp, file)

stp = open('./bangla_stopwords.pkl', 'rb')
stp = pickle.load(stp)
len(stp)

data['Length'] = data.Cleaned_data.apply(lambda x: len(x.split()))

# remove doucment less that 5 length
dataset = data.loc[data.Length > 10]
dataset = dataset.reset_index(drop=True)

# store trained data
dataset.to_csv('trained_dataset.csv')

dataset = pd.read_csv('trained_dataset.csv')
dataset.columns


def data_summary(dataset):
    documents = []
    words = []
    u_words = []
    class_label = [
        k for k, v in dataset.Category.value_counts().to_dict().items()]
    for label in class_label:
        word_list = [word.strip().lower() for t in list(
            dataset[dataset.Category == label].Cleaned_data) for word in t.strip().split()]
        counts = dict()
        for word in word_list:
            counts[word] = counts.get(word,0)+1
        ordered = sorted(counts.items() ,key=lambda item:item[1],reverse=True)

        documents.append(len(list(dataset[dataset.Category == label].Cleaned_data)))

        words.append(len(word_list))

        u_words.append(len(np.unique(word_list)))

        print('\n Class Name : ',label)
        print('Number of Documents : {}'.format(len(list(dataset[dataset.Category ==label].Cleaned_data))))
        for k, v in ordered[:10]:
            print("{} \t {}".format(k,v))

    return documents , words,u_words,class_label


documents, words, u_words, class_names = data_summary(dataset)


# level encoding and dataset splitting

def label_encoding(dataset,bool):
    le = LabelEncoder()
    le.fit(dataset.Category)
    encoded_labels = le.transform(dataset.Category)
    labels = np.array(encoded_labels)
    class_names = le.classes_
    if bool == True:
        print("aa")
    return labels

def dataset_split(news,category):
    print('news')
    print(category)
    X, X_test, y, y_test = train_test_split(news, category, train_size=0.9,test_size=0.1, random_state=0)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=0)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

#Tokenizer


def encoded_texts(dataset,padding_length,max_words):
    tokenizer = Tokenizer(num_words = max_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n-',split =' ',char_level =False,oov_token='<oov>',document_count=0)
    
    tokenizer.fit_on_texts(dataset.Cleaned_data)

    (word_counts,word_docs,word_index,document_count) = (tokenizer.word_counts,tokenizer.word_docs,tokenizer.word_index,tokenizer.document_count)

    def tokenizer_info(mylist,bool):
        ordered = sorted(mylist.items(),key=lambda item:item[1],reverse=bool)
        for w,c in ordered[:10]:
            print(w,"\t",c)

    tokenizer_info(word_counts,bool=True)

    tokenizer_info(word_docs,bool= True)

    tokenizer_info(word_index,bool=True)

    # convert string into list of int

    sequences = tokenizer.texts_to_sequences(dataset.Cleaned_data)
    word_index = tokenizer.word_index

    corpus = keras.preprocessing.sequence.pad_sequences(sequences,value=0.0, padding = 'post' ,maxlen = padding_length)

    labels = label_encoding(dataset, True)

    with open('tokenizer.pickle', 'wb')  as handle:
        pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
    return corpus,labels
    

num_words = 5000
corpus,labels = encoded_texts(dataset, 300, num_words)

print(corpus)
print(labels)

dataset_split(corpus, labels)

#define model

embedding_dimension = 128
input_length = 300
vocab_size = 5000
num_classes = 12
batch_size = 64
num_epochs = 10
accuracy_threshold = 0.97


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,log={}):
        if (logs.get('accuracy') > accuracy_threshold):
            print("\nReached %2.2f%% accuracy so we will stop trianing" % (accuracy_threshold*100))
            self.model.stop_training = True
acc_callback = myCallback()

#save model

filepath = 'Model.h5'

    

