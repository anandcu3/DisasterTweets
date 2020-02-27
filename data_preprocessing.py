import numpy as np
import pandas as pd
import re
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


cleaned_data = preprocess()


