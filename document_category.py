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

#cleaning data remove symbols

def data_cleaning(articles):
    articles_data_clean = articles.replace('\n',' ')
    articles_data_clean = re.sub(
        '[^\u0980-\u09FF]', ' ', str(articles_data_clean))
    stop_words = open('./bangla_stopwords.txt', 'r', encoding='utf-8').read().split()
    result = articles_data_clean.split()
    articles_data_clean = [word.strip() for word in result if word not in stop_words]
    articles_data_clean = " ".join(articles_data_clean)
    return articles_data_clean

data['Cleaned_data'] = data['Text'].apply(data_cleaning)

sample_data = [0]

# for i in sample_data:
#     print('Orginal:\n',data.Text[i])
#     print('Orginal:\n', data.Cleaned_data[i])
#     print('Orginal:\n',data.Category[i])

data['Length'] = data.Cleaned_data.apply(lambda x: len(x.split()))

#remove doucment less that 5 length
dataset = data.loc[data.Length>10]
dataset = dataset.reset_index(drop = True)

# store trained data
dataset.to_csv('trained_dataset.csv')

dataset = pd.read_csv('trained_dataset.csv')
dataset.columns
