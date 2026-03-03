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
                 encoder_max_seq_len,
                 vocab_file):
        super().__init__()
        self.decoder_stack_height = decoder_stack_height
        self.d_model = d_model
        self.h_model = h_model
        self.decoder_dropout_rate = decoder_dropout_rate
        self.n_decoder_vocab = n_decoder_vocab
        self.label_seq_length = label_seq_length
        self.encoder_max_seq_len = encoder_max_seq_len
        self.vocab_file = vocab_file
        self.encoder = DistilBertEncoder(self.d_model, self.vocab_file, self.encoder_max_seq_len)
        self.decoder = MyCustomSimpleDecoder(self.d_model,
                                             self.decoder_stack_height,
                                             self.d_model,
                                             self.h_model,
                                             self.decoder_dropout_rate)
        self.embedding = layers.Embedding(self.n_decoder_vocab+1, self.d_model)
        self.positional_encoding = PositionalEncoder(self.d_model)
        self.output_layer = layers.Dense(units=self.n_decoder_vocab+1)
        self.output_seq_length = self.label_seq_length

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decoder_stack_height ": self.decoder_stack_height,
                "d_model": self.d_model,
                "h_model": self.h_model,
                "decoder_dropout_rate": self.decoder_dropout_rate,
                "n_decoder_vocab": self.n_decoder_vocab,
                "label_seq_length": self.label_seq_length,
                "encoder_max_seq_len": self.encoder_max_seq_len,
                "vocab_file": self.vocab_file
            }
        )
        return config


class MyCustomModel1(keras.Model):
    def __init__(self,
                 decoder_stack_height,
                 d_keys, d_values, h_model,
                 decoder_dropout_rate,
                 n_decoder_vocab,
                 label_seq_length,
                 encoder_max_seq_len,
                 vocab_file):
        super().__init__()
        self.decoder_stack_height = decoder_stack_height
        self.d_keys = d_keys
        self.d_values = d_values
        self.h_model = h_model
        self.decoder_dropout_rate = decoder_dropout_rate
        self.n_decoder_vocab = n_decoder_vocab
        self.label_seq_length = label_seq_length
        self.encoder_max_seq_len = encoder_max_seq_len
        self.vocab_file = vocab_file

        self.encoder = DistilBertEncoder(
            self.d_values, self.vocab_file, self.encoder_max_seq_len)
        self.decoder = MyCustomDecoder(self.d_values,
                                       self.decoder_stack_height,
                                       self.d_keys,
                                       self.d_values,
                                       self.h_model,
                                       self.decoder_dropout_rate)
        self.embedding = layers.Embedding(
            self.n_decoder_vocab+1, self.d_values)
        self.positional_encoding = PositionalEncoder(self.d_values)
        # self.output_layer = layers.Dense(units=n_decoder_vocab+1)
        self.output_layer = layers.Dense(units=1)
        self.output_seq_length = self.label_seq_length

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decoder_stack_height ": self.decoder_stack_height,
                "d_keys": self.d_keys,
                "d_values": self.d_values,
                "h_model": self.h_model,
                "decoder_dropout_rate": self.decoder_dropout_rate,
                "n_decoder_vocab": self.n_decoder_vocab,
                "label_seq_length": self.label_seq_length,
                "encoder_max_seq_len": self.encoder_max_seq_len,
                "vocab_file": self.vocab_file
            }
        )
        return config
