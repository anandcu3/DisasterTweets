import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import ssl


"these check if gpu is available or not"
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#bert tokenization file
import tokenization

#prepare bert embedding format for dataset
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        # subtract 2 as cls ans sep tags to be added
        text = text[:max_len - 2]
        # get text to comply with bert embedding format
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        #print("Input: ", input_sequence)
        pad_len = max_len - len(input_sequence)
        #print("Input pad len :", pad_len)

        # assigns each word a token number and converts text to token ids
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        #print("Token ids : ", tokens)

        # padding till 160 length where 0 is the default
        tokens += [0] * pad_len
        #print("Tokens after padding : ", tokens)

        # padding 1 and 0 where 1 tells token present
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        #print("Pad masks :", pad_masks)

        # array of only 0's
        segment_ids = [0] * max_len
        #print("Segments : ", segment_ids)

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    #print("Input mask : ", input_mask)

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    #print("segments : ", segment_ids)

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    #print("Seq Output: ", sequence_output)

    clf_output = sequence_output[:, 0, :]
    #print("CLF output: ", clf_output)
    out = Dense(1, activation='sigmoid')(clf_output)
    #print("After activation: ", out)

    model = Model(inputs=[input_word_ids, input_mask,
                          segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


ssl._create_default_https_context = ssl._create_default_https_context
import os
os.environ["PYTHONHTTPSVERIFY"] = "0"
module_url = "1"
#module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):

    bert_layer = hub.KerasLayer(module_url, trainable=True)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    test_input = bert_encode(test.text.values, tokenizer, max_len=160)
    train_labels = train.target.values

    model = build_model(bert_layer, max_len=160)
    model.summary()

    train_history = model.fit(
        train_input, train_labels,
        validation_split=0.2,
        epochs=15,
        batch_size=4
    )

    model.save('model_new.h5')