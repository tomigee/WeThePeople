import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm import tqdm

from .custom_layers import MyCustomDecoder, PositionalEncoder, DistilBertEncoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class MyCustomModel(keras.Model):
    def __init__(self,
                 decoder_stack_height,
                 d_model, h_model,
                 decoder_dropout_rate,
                 n_decoder_vocab,
                 label_seq_length):
        super().__init__()
        self.encoder = DistilBertEncoder(d_model)
        self.decoder = MyCustomDecoder(d_model,
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
        # creates embedding values for each item in the list. Output shae: [Ty, dv1]
        dec_inp = self.embedding(prebill_series)
        # not efficient cuz context vectors recomputed every time
        for i in tqdm(range(self.output_seq_length)):
            # augments embedding id's with positional data. Output shape: [Ty, dv1]
            x = dec_inp + self.positional_encoding(dec_inp)
            x = self.decoder(x, context)
            # Get last item
            new_token_emb = tf.expand_dims(x[:, -1, :], 1)
            dec_inp = tf.concat([dec_inp, new_token_emb], 1)

        x = dec_inp[:, -self.output_seq_length:, :]
        x = self.output_layer(x)

        return x
