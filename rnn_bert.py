
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation, SpatialDropout1D, Dropout, Input
from tensorflow.keras.models import Model
import tensorflow as tf


def build_model(embedding_layer, max_len):
    # create Model
    input_word_ids = Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    #print("Input mask : ", input_mask)

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    #print("segments : ", segment_ids)

    _, sequence_output = embedding_layer(
        [input_word_ids, input_mask, segment_ids])
    #print("Seq Output: ", sequence_output)

    clf_output = tf.expand_dims(sequence_output[:, 0, :], 2)
    print("CLF output: ", clf_output.shape)

    out = LSTM(32)(clf_output)

    out = Dense(1)(out)
    #print("After activation: ", out)

    model = Model(inputs=[input_word_ids, input_mask,
                          segment_ids], outputs=out)
    model.compile(Adam(lr=1e-4), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
