import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import os
import nltk

# nltk.download() uncomment this if not downloaded
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn import model_selection


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam

def read():
    data_train = pd.read_csv('data/train.csv')
    print("Data shape = ", data_train.shape)
    data_train.head()
    return data_train

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'' , text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'' , text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

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
    #print(data_train.columns)

    cleaned_data = []
    target_labels = []
    porterstem = PorterStemmer()

    for i in range(data_train.shape[0]):
        # Remove unwanted words

        target = data_train.target[i]

        tweet_text = re.sub("[^a-zA-Z]", ' ', data_train['text'][i])
        # Transform words to lowercase
        tweet_text = tweet_text.lower()
        tweet_text = tweet_text.split()
        # Remove stopwords then Stemming it
        tweet_text = [porterstem.stem(word) for word in tweet_text if not word in set(stopwords.words('english'))]
        tweet_text = ' '.join(tweet_text)

        tweet_text = remove_URL(tweet_text)
        tweet_text = remove_html(tweet_text)
        tweet_text = remove_punct(tweet_text)
        tweet_text = remove_emoji(tweet_text)

        # Append cleaned tweet to corpus
        cleaned_data.append(tweet_text)
        target_labels.append(target)
    print("Corpus created successfully")
    print(pd.DataFrame(cleaned_data)[0].head(10))

    return cleaned_data, target_labels

def create_corpus_new(df):
    corpus=[]
    for tweet in tqdm(df):
        words=[word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus

cleaned_data, target_labels = preprocess()
corpus=create_corpus_new(cleaned_data)

embedding_dict={}
file_name = "data/glove.twitter.27B.100d.txt"
with open(file_name,'r', encoding="utf8") as f:
    for line in f:
        values=line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()


MAX_LEN=50  #assuming max 50 similar words
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
train_tweets = pd.read_csv('data/train.csv')
train_labels = train_tweets.target.values

train=tweet_pad[:train_tweets.shape[0]]
#print("Train padded: ", train)

word_index=tokenizer_obj.word_index

num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

#create glove embedding matrix
for word,i in tqdm(word_index.items()):
    if i < num_words:
        emb_vec=embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i]=emb_vec

#print sample output

print(f'embedding matrix shape -> {embedding_matrix.shape}')
print(f'example: word \'tornado\' has index of -> {word_index["tornado"]}')
print(f'{embedding_matrix[word_index["tornado"]]}')


#create Model
model=Sequential()
embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

#Simple NN model
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
optimzer=Adam(learning_rate=3e-4)
model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()

#Data
X_train,X_test,y_train,y_test=model_selection.train_test_split(train,train_labels,test_size=0.2)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)

#model fitting
history=model.fit(X_train,y_train,batch_size=4,epochs=10,validation_data=(X_test,y_test),verbose=2)