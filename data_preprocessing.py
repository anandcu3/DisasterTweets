import numpy as np
import pandas as pd
import re
import string
import seaborn as sns
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import operator
import os
import nltk
#nltk.download() uncomment this if not downloaded
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

### this script will do the preprocessing for training data

def read():
    data_train = pd.read_csv('data/train.csv')
    print("Data shape = ", data_train.shape)
    data_train.head()
    return data_train


def preprocess():
    """
        Remove unwanted words
        Transform words to lowercase
        Remove stopwords
        Stemming words
        """
    data_train = read()
    # handle missing data and drop columns not needed. location and keyword are giving misleading information.
    data_train = data_train.drop(['location', 'keyword', 'id'], axis=1)
    print(data_train.columns)

    cleaned_data = []
    porterstem = PorterStemmer()
    for i in range(data_train['text'].shape[0]):
        # Remove unwanted words
        tweet_text = re.sub("[^a-zA-Z]", ' ', data_train['text'][i])
        # Transform words to lowercase
        tweet_text = tweet_text.lower()
        tweet_text = tweet_text.split()
        # Remove stopwords then Stemming it
        tweet_text = [porterstem.stem(word) for word in tweet_text if not word in set(stopwords.words('english'))]
        tweet_text = ' '.join(tweet_text)
        # Append cleaned tweet to corpus
        cleaned_data.append(tweet_text)

    print("Corpus created successfully")
    print(pd.DataFrame(cleaned_data)[0].head(10))

    return cleaned_data

def zipfs_law_plot():
    cleaned_data = preprocess()
    frequency_dict = {}
    for tweets in cleaned_data:
        for word in tweets.split():
            count = frequency_dict.get(word, 0)
            frequency_dict[word] = count + 1

    sorted_doc = (sorted(frequency_dict.items(), key=operator.itemgetter(1)))[::-1]
    frequency = []
    rank = []

    entry_num = 1
    for entry in sorted_doc:
        rank.append(entry_num)
        entry_num += 1
        frequency.append(entry[1])

    # calculates slope and intercept value ( y = mx + c)
    m, c = np.polyfit(np.log(rank), np.log(frequency), 1)
    y_fit = np.exp(m * np.log(rank) + c)

    plt.loglog(frequency, label='Dataset')
    plt.loglog(y_fit, ':', label='Zipf')
    plt.xlabel('rank')
    plt.ylabel('frequency')
    plt.title("Word frequencies, Zipf's law")
    plt.legend()
    plt.show()
    sns.despine(trim=True)

zipfs_law_plot()

