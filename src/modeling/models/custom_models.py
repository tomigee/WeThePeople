import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm import tqdm

from .custom_layers import (
    MyCustomSimpleDecoder,
    PositionalEncoder,
    DistilBertEncoder,
    MyCustomDecoder
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class MyCustomModel(keras.Model):
    def __init__(self,
                 decoder_stack_height,
                 d_model, h_model,
                 decoder_dropout_rate,
                 n_decoder_vocab,
                 label_seq_length,
                 encoder_max_seq_len):
        super().__init__()
        self.encoder = DistilBertEncoder(d_model, encoder_max_seq_len)
        self.decoder = MyCustomSimpleDecoder(d_model,
                                             decoder_stack_height,
                                             d_model,
                                             h_model,
                                             decoder_dropout_rate)
        self.embedding = layers.Embedding(n_decoder_vocab+1, d_model)
        self.positional_encoding = PositionalEncoder(d_model)
        self.output_layer = layers.Dense(units=n_decoder_vocab+1)
        self.output_seq_length = label_seq_length

    def call(self, input):
        # Input is a tuple
        bill_text, prebill_series = input
        context = self.encoder(bill_text)

        # Input is a list of ids (tokenized)
        # creates embedding values for each item in the list. Output shae: [batch, Ty, dv1]
        dec_inp = self.embedding(prebill_series)  # decoder input
        # not efficient cuz context vectors recomputed every time
        for i in tqdm(range(self.output_seq_length)):
            # augments embedding id's with positional data. Output shape: [batch, Ty, dv1]
            x = dec_inp + self.positional_encoding(dec_inp)
            x = self.decoder(x, context)
            # Get last item
            new_token_emb = tf.expand_dims(x[:, -1, :], 1)
            dec_inp = tf.concat([dec_inp, new_token_emb], 1)

        x = dec_inp[:, -self.output_seq_length:, :]
        x = self.output_layer(x)

        return x


class MyCustomModel1(keras.Model):
    def __init__(self,
                 decoder_stack_height,
                 d_keys, d_values, h_model,
                 decoder_dropout_rate,
                 n_decoder_vocab,
                 label_seq_length,
                 encoder_max_seq_len):
        super().__init__()
        self.encoder = DistilBertEncoder(d_values, encoder_max_seq_len)
        self.decoder = MyCustomDecoder(d_values,
                                       decoder_stack_height,
                                       d_keys,
                                       d_values,
                                       h_model,
                                       decoder_dropout_rate)
        self.embedding = layers.Embedding(n_decoder_vocab+1, d_values)
        self.positional_encoding = PositionalEncoder(d_values)
        # self.output_layer = layers.Dense(units=n_decoder_vocab+1)
        self.output_layer = layers.Dense(units=1)
        self.output_seq_length = label_seq_length

    def call(self, input):
        # Input is a tuple
        bill_text, prebill_series = input
        context = self.encoder(bill_text)

        # Input is a list of ids (tokenized)
        # creates embedding values for each item in the list. Output shae: [batch, Ty, dv1]
        dec_inp = self.embedding(prebill_series)  # decoder input
        # not efficient cuz context vectors recomputed every time
        for i in tqdm(range(self.output_seq_length)):
            # augments embedding id's with positional data. Output shape: [batch, Ty, dv1]
            x = dec_inp + self.positional_encoding(dec_inp)
            x = self.decoder(x, context)
            # Get last item
            new_token_emb = tf.expand_dims(x[:, -1, :], 1)
            dec_inp = tf.concat([dec_inp, new_token_emb], 1)

        x = dec_inp[:, -self.output_seq_length:, :]
        x = self.output_layer(x)

        return x
