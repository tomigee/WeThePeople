import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm import tqdm

from .custom_layers import (
    MyCustomSimpleDecoder,
    PositionalEncoder,
    DistilBertEncoder,
    MyCustomDecoder,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class MyCustomModel(keras.Model):
    """Encoder-Decoder Transformer that predicts numerical time-series based on text input.

    Args:
        decoder_stack_height (int): Number of decoder blocks in decoder
        d_model (int): Characteristic vector length used for configuring attention layers.
        h_model (int): Number of heads for all Multi Head Attention blocks in the decoder stack.
        decoder_dropout_rate (float): Characteristic dropout rate used across decoder stack.
        n_decoder_vocab (int): Number of entries in decoder (numerical) vocabulary.
        label_seq_length (int): Length of time-series ultimately predicted by decoder.
        encoder_max_seq_len (int): Length to which the encoder trims tokenized text sequences.
        vocab_file (_type_): Location of vocab file in filesystem.
    """

    def __init__(
        self,
        decoder_stack_height: int,
        d_model: int,
        h_model: int,
        decoder_dropout_rate: float,
        n_decoder_vocab: int,
        label_seq_length: int,
        encoder_max_seq_len: int,
        vocab_file: str,
    ):
        super().__init__()
        self.decoder_stack_height = decoder_stack_height
        self.d_model = d_model
        self.h_model = h_model
        self.decoder_dropout_rate = decoder_dropout_rate
        self.n_decoder_vocab = n_decoder_vocab
        self.label_seq_length = label_seq_length
        self.encoder_max_seq_len = encoder_max_seq_len
        self.vocab_file = vocab_file
        self.encoder = DistilBertEncoder(
            self.d_model, self.vocab_file, self.encoder_max_seq_len
        )
        self.decoder = MyCustomSimpleDecoder(
            self.d_model,
            self.decoder_stack_height,
            self.d_model,
            self.h_model,
            self.decoder_dropout_rate,
        )
        self.embedding = layers.Embedding(self.n_decoder_vocab + 1, self.d_model)
        self.positional_encoding = PositionalEncoder(self.d_model)
        self.output_layer = layers.Dense(units=self.n_decoder_vocab + 1)
        self.output_seq_length = self.label_seq_length

    def call(self, input: tuple[tf.data.Dataset, tf.data.Dataset]) -> tf.Tensor:
        """Forward propagation."""
        bill_text, prebill_series = input
        context = self.encoder(bill_text)
        dec_inp = self.embedding(
            prebill_series
        )  # Output shape: [batch, Ty, dv1], decoder input
        # not efficient cuz context vectors recomputed every time
        for i in tqdm(range(self.output_seq_length)):
            x = dec_inp + self.positional_encoding(
                dec_inp
            )  # Output shape: [batch, Ty, dv1]
            x = self.decoder(x, context)
            new_token_emb = tf.expand_dims(x[:, -1, :], 1)  # Get last item
            dec_inp = tf.concat([dec_inp, new_token_emb], 1)

        x = dec_inp[:, -self.output_seq_length :, :]
        x = self.output_layer(x)

        return x

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
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
                "vocab_file": self.vocab_file,
            }
        )
        return config


class MyCustomModel1(keras.Model):
    """Encoder-Decoder Transformer that predicts numerical time-series based on text input.

    Args:
        decoder_stack_height (int): Number of decoder blocks in decoder
        d_keys (int): Length of key vectors in all attention layers except within BERT pretrained layer.
        d_values (int): Length of value vectors in all attention layers except within BERT pretrained layer.
        h_model (int): Number of heads for all Multi Head Attention blocks in the decoder stack.
        decoder_dropout_rate (float): Characteristic dropout rate used across decoder stack.
        n_decoder_vocab (int): Number of entries in decoder (numerical) vocabulary.
        label_seq_length (int): Length of time-series ultimately predicted by decoder.
        encoder_max_seq_len (int): Length to which the encoder trims tokenized text sequences.
        vocab_file (_type_): Location of vocab file in filesystem.
    """  # noqa: E501

    def __init__(
        self,
        decoder_stack_height: int,
        d_keys: int,
        d_values: int,
        h_model: int,
        decoder_dropout_rate: float,
        n_decoder_vocab: int,
        label_seq_length: int,
        encoder_max_seq_len: int,
        vocab_file: str,
    ):
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
            self.d_values, self.vocab_file, self.encoder_max_seq_len
        )
        self.decoder = MyCustomDecoder(
            self.d_values,
            self.decoder_stack_height,
            self.d_keys,
            self.d_values,
            self.h_model,
            self.decoder_dropout_rate,
        )
        self.embedding = layers.Embedding(self.n_decoder_vocab + 1, self.d_values)
        self.positional_encoding = PositionalEncoder(self.d_values)
        self.output_layer = layers.Dense(units=1)
        self.output_seq_length = self.label_seq_length

    def call(self, input: tuple[tf.data.Dataset, tf.data.Dataset]) -> tf.Tensor:
        """Forward propagation."""
        bill_text, prebill_series = input
        context = self.encoder(bill_text)
        dec_inp = self.embedding(
            prebill_series
        )  # Output shape: [batch, Ty, dv1], decoder input
        # not efficient cuz context vectors recomputed every time
        for i in tqdm(range(self.output_seq_length)):
            x = dec_inp + self.positional_encoding(
                dec_inp
            )  # Output shape: [batch, Ty, dv1]
            x = self.decoder(x, context)
            new_token_emb = tf.expand_dims(x[:, -1, :], 1)  # Get last item
            dec_inp = tf.concat([dec_inp, new_token_emb], 1)

        x = dec_inp[:, -self.output_seq_length :, :]
        x = self.output_layer(x)

        return x

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
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
                "vocab_file": self.vocab_file,
            }
        )
        return config
