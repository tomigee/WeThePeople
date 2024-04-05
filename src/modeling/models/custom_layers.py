import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_text as tfText

from transformers import TFBertModel, TFDistilBertModel
import numpy as np


class PositionalEncoder(layers.Layer):
    def __init__(self, output_dim):
        # output_dim: dimensionality of positional encoding vector
        super().__init__()
        self.output_dim = output_dim

    def call(self, inputs):
        entry = np.tile(np.expand_dims(
            np.arange(inputs.shape[1]), -1), (1, self.output_dim))
        two_i = np.tile(
            np.repeat(np.arange(0, self.output_dim, 2), 2), (inputs.shape[1], 1))
        if not self.output_dim % 2 == 0:
            two_i = two_i[:, :-1]
        base = 10000*np.ones([inputs.shape[1], self.output_dim])
        quotient = entry/(np.power(base, (two_i/self.output_dim)))
        sin_mask = np.tile(np.arange(self.output_dim),
                           (inputs.shape[1], 1)) % 2 == 0
        cos_mask = np.logical_not(sin_mask)
        output = sin_mask*np.sin(quotient) + cos_mask*np.cos(quotient)
        return output.astype("float32")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
            }
        )
        return config


class BaseAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, value_dim, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout_rate = dropout_rate

        self.add = layers.Add()
        self.norm = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(
            self.num_heads, self.key_dim, self.value_dim, dropout=self.dropout_rate)
    # add a build function for mha (per docs)?

    def build(self, inputs):
        # super().build()
        self.mha._build_from_signature(query=inputs, value=inputs, key=inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "dropout_rate": self.dropout_rate
            }
        )
        return config


class SimpleSelfAttention(BaseAttention):
    def call(self, inputs):
        x = self.mha(query=inputs, value=inputs, key=inputs)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x


class SimpleCrossAttention(BaseAttention):
    def call(self, inputs, context):
        x = self.mha(query=inputs, value=context, key=context)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x


class SimpleCausalSelfAttention(BaseAttention):
    def call(self, inputs):
        x = self.mha(query=inputs, value=inputs,
                     key=inputs, use_causal_mask=True)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x


class FeedForwardNN(layers.Layer):
    def __init__(self, output_dim, ff_dropout_rate):
        super().__init__()
        self.output_dim = output_dim
        self.ff_dropout_rate = ff_dropout_rate

        self.relu = layers.Dense(units=self.output_dim, activation='relu')
        self.linear = layers.Dense(units=self.output_dim)
        self.dropout = layers.Dropout(self.ff_dropout_rate)
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()

    def call(self, inputs):
        x = self.relu(inputs)
        x = self.linear(x)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "ff_dropout_rate": self.ff_dropout_rate
            }
        )
        return config


class MyCustomDecoderLayer(layers.Layer):
    def __init__(self,
                 output_dim,
                 sa_num_heads,
                 ca_num_heads,
                 sa_key_dim,
                 ca_key_dim,
                 sa_value_dim,
                 ca_value_dim,
                 ca_dropout_rate,
                 ff_dropout_rate):
        super().__init__()
        self.output_dim = output_dim
        self.sa_num_heads = sa_num_heads
        self.ca_num_heads = ca_num_heads
        self.sa_key_dim = sa_key_dim
        self.ca_key_dim = ca_key_dim
        self.sa_value_dim = sa_value_dim
        self.ca_value_dim = ca_value_dim
        self.ca_dropout_rate = ca_dropout_rate
        self.ff_dropout_rate = ff_dropout_rate
        # masked self attention, no dropout
        self.msa = SimpleCausalSelfAttention(
            self.sa_num_heads, self.sa_key_dim, self.sa_value_dim, 0.0)
        self.ca = SimpleCrossAttention(self.ca_num_heads,
                                       self.ca_key_dim,
                                       self.ca_value_dim,
                                       self.ca_dropout_rate)  # cross attention
        self.ffn = FeedForwardNN(self.output_dim, self.ff_dropout_rate)

    def call(self, inputs, context):
        x = self.msa(inputs)
        x = self.ca(x, context)  # x returned here is dim 203
        x = self.ffn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "sa_num_heads": self.sa_num_heads,
                "ca_num_heads": self.ca_num_heads,
                "sa_key_dim": self.sa_key_dim,
                "ca_key_dim": self.ca_key_dim,
                "sa_value_dim": self.sa_value_dim,
                "ca_value_dim": self.ca_value_dim,
                "ca_dropout_rate": self.ca_dropout_rate,
                "ff_dropout_rate": self.ff_dropout_rate
            }
        )
        return config


class MyCustomSimpleDecoder(layers.Layer):
    def __init__(self, output_dim, stack_height, d_model, h_model, dropout_rate):
        super().__init__()
        self.output_dim = output_dim
        self.stack_height = stack_height
        self.d_model = d_model
        self.h_model = h_model
        self.dropout_rate = dropout_rate
        self.decoder_layers = [MyCustomDecoderLayer(self.output_dim,
                                                    self.h_model,
                                                    self.h_model,
                                                    self.d_model,
                                                    self.d_model,
                                                    int(self.d_model /
                                                        self.h_model),
                                                    int(self.d_model /
                                                        self.h_model),
                                                    self.dropout_rate,
                                                    self.dropout_rate)
                               for _ in range(self.stack_height)]

    def call(self, input, context):
        x = input
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, context)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "stack_height": self.stack_height,
                "d_model": self.d_model,
                "h_model": self.h_model,
                "dropout_rate": self.dropout_rate
            }
        )
        return config


class MyCustomDecoder(layers.Layer):
    def __init__(self, output_dim, stack_height, d_keys, d_values, h_model, dropout_rate):
        super().__init__()
        self.output_dim = output_dim
        self.stack_height = stack_height
        self.d_keys = d_keys
        self.d_values = d_values
        self.h_model = h_model
        self.dropout_rate = dropout_rate

        self.decoder_layers = [MyCustomDecoderLayer(self.output_dim,
                                                    self.h_model,
                                                    self.h_model,
                                                    self.d_keys,
                                                    self.d_keys,
                                                    self.d_values,
                                                    self.d_values,
                                                    self.dropout_rate,
                                                    self.dropout_rate)
                               for _ in range(self.stack_height)]

    def call(self, input, context):
        x = input
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, context)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "stack_height": self.stack_height,
                "d_keys": self.d_keys,
                "d_values": self.d_values,
                "h_model": self.h_model,
                "dropout_rate": self.dropout_rate
            }
        )
        return config


class MyBertTokenizer(layers.Layer):
    START_TOKEN = 101
    END_TOKEN = 102

    def __init__(self):
        super().__init__()
        self.tokenizer = tfText.BertTokenizer("datasets/Bert_Vocabulary.txt")

    def call(self, inputs):
        tokenized = self.tokenizer.tokenize(
            tf.strings.lower(inputs)).merge_dims(-2, -1)
        processed_segments, segment_ids = tfText.combine_segments([tokenized],
                                                                  MyBertTokenizer.START_TOKEN,
                                                                  MyBertTokenizer.END_TOKEN)
        processed_segments = tf.cast(
            processed_segments.to_tensor(), dtype=tf.int32)
        segment_ids = tf.cast(segment_ids.to_tensor(), dtype=tf.int32)
        return {'input_ids': processed_segments,
                'token_type_ids': segment_ids}


class MyBertTokenizerTrimmed(layers.Layer):
    START_TOKEN = 101
    END_TOKEN = 102

    def __init__(self, max_seq_len, vocab_file):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_file = vocab_file
        self.tokenizer = tfText.BertTokenizer(self.vocab_file)
        self.trimmer = tfText.RoundRobinTrimmer(
            max_seq_length=self.max_seq_len)

    def call(self, inputs):
        tokenized = self.tokenizer.tokenize(
            tf.strings.lower(inputs)).merge_dims(-2, -1)
        trimmed = self.trimmer.trim([tokenized])
        processed_segments, segment_ids = tfText.combine_segments(trimmed,
                                                                  MyBertTokenizer.START_TOKEN,
                                                                  MyBertTokenizer.END_TOKEN)
        processed_segments = tf.cast(
            processed_segments.to_tensor(), dtype=tf.int32)
        segment_ids = tf.cast(segment_ids.to_tensor(), dtype=tf.int32)
        return {'input_ids': processed_segments,
                'token_type_ids': segment_ids}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_seq_len": self.max_seq_len,
                "vocab_file": self.vocab_file
            }
        )
        return config


class BertEncoder(layers.Layer):
    def __init__(self, projection_dim, max_seq_length=None):
        super().__init__()
        self.projection_dim = projection_dim
        self.max_seq_length = max_seq_length
        self.tokenizer = MyBertTokenizer()
        if self.max_seq_length:
            self.bert = TFBertModel.from_pretrained(
                "google-bert/bert-base-uncased",
                max_position_embeddings=self.max_seq_length,
                ignore_mismatched_sizes=True
            )
        else:
            self.bert = TFBertModel.from_pretrained(
                "google-bert/bert-base-uncased")

        self.bert.trainable = False  # freeze pre-trained weights
        self.broadcaster = layers.Dense(units=self.projection_dim)

    def call(self, input):
        x = self.tokenizer(input)
        x = self.bert(**x)
        x = tf.expand_dims(self.broadcaster(x.pooler_output), 1)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "max_seq_length": self.max_seq_length
            }
        )
        return config


class DistilBertEncoder(layers.Layer):
    def __init__(self, projection_dim, vocab_file, max_seq_length=None):
        super().__init__()
        self.projection_dim = projection_dim
        self.max_seq_length = max_seq_length
        self.vocab_file = vocab_file

        if self.max_seq_length:
            self.tokenizer = MyBertTokenizerTrimmed(
                self.max_seq_length, self.vocab_file)
            self.bert = TFDistilBertModel.from_pretrained(
                "distilbert/distilbert-base-uncased",
                max_position_embeddings=self.max_seq_length,
                ignore_mismatched_sizes=True
            )
        else:
            self.tokenizer = MyBertTokenizer()
            self.bert = TFDistilBertModel.from_pretrained(
                "distilbert/distilbert-base-uncased")

        self.bert.trainable = False  # freeze pre-trained weights
        self.broadcaster = layers.Dense(units=self.projection_dim)

    def call(self, input):
        x = self.tokenizer(input)
        x.pop("token_type_ids")
        x = self.bert(**x)
        x = self.broadcaster(x.last_hidden_state)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "vocab_file": self.vocab_file,
                "max_seq_length": self.max_seq_length
            }
        )
        return config
