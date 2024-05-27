import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_text as tfText

from transformers import TFBertModel, TFDistilBertModel
import numpy as np


class PositionalEncoder(layers.Layer):
    """Positional encoder, as seen in `Attention Is All You Need`

    Attributes:
        output_dim (int): Length of resulting positional encoding vector
    """

    def __init__(self, output_dim: int):
        """Initializes instance based on desired positional encoding vector length

        Args:
            output_dim (int): Length of resulting positional encoding vector
        """
        super().__init__()
        self.output_dim = output_dim

    def call(self, inputs: tf.Tensor) -> np.ndarray:
        """Forward propagation."""
        entry = np.tile(
            np.expand_dims(np.arange(inputs.shape[1]), -1), (1, self.output_dim)
        )
        two_i = np.tile(
            np.repeat(np.arange(0, self.output_dim, 2), 2), (inputs.shape[1], 1)
        )
        if not self.output_dim % 2 == 0:
            two_i = two_i[:, :-1]
        base = 10000 * np.ones([inputs.shape[1], self.output_dim])
        quotient = entry / (np.power(base, (two_i / self.output_dim)))
        sin_mask = np.tile(np.arange(self.output_dim), (inputs.shape[1], 1)) % 2 == 0
        cos_mask = np.logical_not(sin_mask)
        output = sin_mask * np.sin(quotient) + cos_mask * np.cos(quotient)
        return output.astype("float32")

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
            }
        )
        return config


class BaseAttention(layers.Layer):
    """Base class for attention block, consisting of add, norm, and multihead attention layers.

    Attributes:
        num_heads (int): Number of heads in Multi Head Attention.
        key_dim (int): Length of the key tensor.
        value_dim (int): Length of the value tensor.
        dropout_rate (float): Dropout rate in Multi Head Attention.
        add (tf.keras.layers.Layer): Adding layer.
        norm (tf.keras.layers.Layer): Normalizing layer.
        mha (tf.keras.layers.Layer): Multi Head Attention layer.
    """

    def __init__(
        self, num_heads: int, key_dim: int, value_dim: int, dropout_rate: float
    ):
        """Initializes instance based on desired attributes."""
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout_rate = dropout_rate

        self.add = layers.Add()
        self.norm = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(
            self.num_heads, self.key_dim, self.value_dim, dropout=self.dropout_rate
        )

    # add a build function for mha (per docs)?

    def build(self, inputs: tf.Tensor) -> None:
        """Not sure if this is necessary in all honesty."""
        # super().build()
        self.mha._build_from_signature(query=inputs, value=inputs, key=inputs)

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class SimpleSelfAttention(BaseAttention):
    """Special case of `BaseAttention` where the query, value, and key are equal. See `BaseAttention` for more details."""  # noqa: E501

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = self.mha(query=inputs, value=inputs, key=inputs)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x


class SimpleCrossAttention(BaseAttention):
    """Special case of `BaseAttention` where the value and key are equal, but different from the query. Generally used in encoder-decoder transformers."""  # noqa: E501

    def call(self, inputs: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = self.mha(query=inputs, value=context, key=context)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x


class SimpleCausalSelfAttention(BaseAttention):
    """Special case of `BaseAttention`. Similar to SimpleSelfAttention, except a causal mask is used in the Multi Head Attention layer, to prevent attention to future tokens."""  # noqa: E501

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = self.mha(query=inputs, value=inputs, key=inputs, use_causal_mask=True)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x


class FeedForwardNN(layers.Layer):
    """Feed Forward Neural Network block consisting of a densely connected NN with ReLU activation, a densely connected NN with no activation, a dropout layer, an add layer, and a normalization layer.

    Attributes:
        output_dim (int): Length of output vector of both Dense layers.
        ff_dropout_rate (float): Dropout rate used in dropout layer.
        relu (tf.keras.layers.Layer): Dense layer, ReLU activation.
        linear (tf.keras.layers.Layer): Dense layer, no activation.
        dropout (tf.keras.layers.Layer): Dropout layer.
        add (tf.keras.layers.Layer): Add layer.
        norm (tf.keras.layers.Layer): Layer normalization layer.
    """  # noqa: E501

    def __init__(self, output_dim: int, ff_dropout_rate: float):
        """Initialize instance based on desired output dimension and dropout rate."""
        super().__init__()
        self.output_dim = output_dim
        self.ff_dropout_rate = ff_dropout_rate

        self.relu = layers.Dense(units=self.output_dim, activation="relu")
        self.linear = layers.Dense(units=self.output_dim)
        self.dropout = layers.Dropout(self.ff_dropout_rate)
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = self.relu(inputs)
        x = self.linear(x)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
        config = super().get_config()
        config.update(
            {"output_dim": self.output_dim, "ff_dropout_rate": self.ff_dropout_rate}
        )
        return config


class MyCustomDecoderLayer(layers.Layer):
    """Decoder block consisting of a `SimpleCausalSelfAttention`, `SimpleCrossAttention`, and `FeedForwardNN` block in sequence.

    Attributes:
        output_dim (int): Length of output vector of the Feed Forward NN.
        sa_num_heads (int): Number of heads in `SimpleCausalSelfAttention` layer.
        ca_num_heads (int): Number of heads in `SimpleCrossAttention` layer.
        sa_key_dim (int): Length of the key tensor in `SimpleCausalSelfAttention` layer.
        ca_key_dim (int): Length of the key tensor in `SimpleCrossAttention` layer.
        sa_value_dim (int): Length of the value tensor in `SimpleCausalSelfAttention` layer.
        ca_value_dim (int): Length of the value tensor in `SimpleCrossAttention` layer.
        ca_dropout_rate (float: Dropout rate in the `SimpleCrossAttention` layer.
        ff_dropout_rate (float): Dropout rate in the `FeedForwardNN` layer.
        msa (tf.keras.layers.Layer): Masked Self Attention block.
        ca (tf.keras.layers.Layer): Cross Attention block.
        ffn (tf.keras.layers.Layer): Feed Forward NN block.
    """  # noqa: E501

    def __init__(
        self,
        output_dim: int,
        sa_num_heads: int,
        ca_num_heads: int,
        sa_key_dim: int,
        ca_key_dim: int,
        sa_value_dim: int,
        ca_value_dim: int,
        ca_dropout_rate: float,
        ff_dropout_rate: float,
    ):
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
            self.sa_num_heads, self.sa_key_dim, self.sa_value_dim, 0.0
        )
        self.ca = SimpleCrossAttention(
            self.ca_num_heads, self.ca_key_dim, self.ca_value_dim, self.ca_dropout_rate
        )  # cross attention
        self.ffn = FeedForwardNN(self.output_dim, self.ff_dropout_rate)

    def call(self, inputs: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = self.msa(inputs)
        x = self.ca(x, context)  # x returned here is dim 203
        x = self.ffn(x)
        return x

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
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
                "ff_dropout_rate": self.ff_dropout_rate,
            }
        )
        return config


class MyCustomSimpleDecoder(layers.Layer):
    """Collection of decoder blocks arranged in sequence.

    Attributes:
        output_dim (int): Length of output vector of the Feed Forward NN.
        stack_height (int): Number of decoder blocks stacked in sequence.
        d_model (int): Characteristic vector length used for configuring attention layers.
        h_model (int): Number of heads for all Multi Head Attention blocks in the decoder stack.
        dropout_rate (float): Characteristic dropout rate used across decoder stack.
        decoder_layers (list[tf.keras.layers.Layer]): Collection of decoder layers in the decoder stack.
    """  # noqa: E501

    def __init__(
        self,
        output_dim: int,
        stack_height: int,
        d_model: int,
        h_model: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.stack_height = stack_height
        self.d_model = d_model
        self.h_model = h_model
        self.dropout_rate = dropout_rate
        self.decoder_layers = [
            MyCustomDecoderLayer(
                self.output_dim,
                self.h_model,
                self.h_model,
                self.d_model,
                self.d_model,
                int(self.d_model / self.h_model),
                int(self.d_model / self.h_model),
                self.dropout_rate,
                self.dropout_rate,
            )
            for _ in range(self.stack_height)
        ]

    def call(self, input: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = input
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, context)
        return x

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "stack_height": self.stack_height,
                "d_model": self.d_model,
                "h_model": self.h_model,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class MyCustomDecoder(layers.Layer):
    """Collection of decoder blocks arranged in sequence.

    Attributes:
        output_dim (int): Length of output vector of the Feed Forward NN.
        stack_height (int): Number of decoder blocks stacked in sequence.
        d_keys (int): Length of key vectors in attention layers.
        d_values (int): Length of value vectors in attention layers.
        h_model (int): Number of heads for all Multi Head Attention blocks in the decoder stack.
        dropout_rate (float): Characteristic dropout rate used across decoder stack.
        decoder_layers (list[tf.keras.layers.Layer]): Collection of decoder layers in the decoder stack.
    """  # noqa: E501

    def __init__(
        self,
        output_dim: int,
        stack_height: int,
        d_keys: int,
        d_values: int,
        h_model: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.stack_height = stack_height
        self.d_keys = d_keys
        self.d_values = d_values
        self.h_model = h_model
        self.dropout_rate = dropout_rate

        self.decoder_layers = [
            MyCustomDecoderLayer(
                self.output_dim,
                self.h_model,
                self.h_model,
                self.d_keys,
                self.d_keys,
                self.d_values,
                self.d_values,
                self.dropout_rate,
                self.dropout_rate,
            )
            for _ in range(self.stack_height)
        ]

    def call(self, input: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = input
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, context)
        return x

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "stack_height": self.stack_height,
                "d_keys": self.d_keys,
                "d_values": self.d_values,
                "h_model": self.h_model,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class MyBertTokenizer(layers.Layer):
    """BERT Tokenizer.

    Attributes:
        vocab_file (str): Location of BERT vocabulary in filesystem.
    """

    START_TOKEN = 101
    END_TOKEN = 102

    def __init__(self, vocab_file: str):
        super().__init__()
        self.vocab_file = vocab_file
        self.tokenizer = tfText.BertTokenizer(self.vocab_file)

    def call(self, inputs: tf.Tensor) -> dict[str, tf.Tensor]:
        """Forward propagation"""
        tokenized = self.tokenizer.tokenize(tf.strings.lower(inputs)).merge_dims(-2, -1)
        processed_segments, segment_ids = tfText.combine_segments(
            [tokenized], MyBertTokenizer.START_TOKEN, MyBertTokenizer.END_TOKEN
        )
        processed_segments = tf.cast(processed_segments.to_tensor(), dtype=tf.int32)
        segment_ids = tf.cast(segment_ids.to_tensor(), dtype=tf.int32)
        return {"input_ids": processed_segments, "token_type_ids": segment_ids}


class MyBertTokenizerTrimmed(layers.Layer):
    """BERT Tokenizer with RoundRobinTrimmer.

    Attributes:
        max_seq_len (int): Length to which tokenized sequences are trimmed.
        vocab_file (str): Location of BERT vocabulary in filesystem.
    """

    START_TOKEN = 101
    END_TOKEN = 102

    def __init__(self, max_seq_len: int, vocab_file: str):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_file = vocab_file
        self.tokenizer = tfText.BertTokenizer(self.vocab_file)
        self.trimmer = tfText.RoundRobinTrimmer(max_seq_length=self.max_seq_len)

    def call(self, inputs: tf.Tensor) -> dict[str, tf.Tensor]:
        """Forward propagation."""
        tokenized = self.tokenizer.tokenize(tf.strings.lower(inputs)).merge_dims(-2, -1)
        trimmed = self.trimmer.trim([tokenized])
        processed_segments, segment_ids = tfText.combine_segments(
            trimmed,
            MyBertTokenizerTrimmed.START_TOKEN,  # noqa: E501
            MyBertTokenizerTrimmed.END_TOKEN,
        )
        processed_segments = tf.cast(processed_segments.to_tensor(), dtype=tf.int32)
        segment_ids = tf.cast(segment_ids.to_tensor(), dtype=tf.int32)
        return {"input_ids": processed_segments, "token_type_ids": segment_ids}

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
        config = super().get_config()
        config.update({"max_seq_len": self.max_seq_len, "vocab_file": self.vocab_file})
        return config


class BertEncoder(layers.Layer):
    """Encoder based on Google's BERT model.

    Attributes:
        projection_dim (_type_): Length of final hidden state vector output by encoder.
        max_seq_length (_type_): Length to which tokenized sequences are trimmed.
        tokenizer (_type_): choice of tokenizer.
        bert (_type_): Pretrained BERT model.
        broadcaster (_type_): Dense layer that broadcasts output of pretrained BERT model into `projection_dim` dimensionality.
    """  # noqa: E501

    def __init__(self, projection_dim: int, max_seq_length: int = None):
        super().__init__()
        self.projection_dim = projection_dim
        self.max_seq_length = max_seq_length
        self.tokenizer = MyBertTokenizer()
        if self.max_seq_length:
            self.bert = TFBertModel.from_pretrained(
                "google-bert/bert-base-uncased",
                max_position_embeddings=self.max_seq_length,
                ignore_mismatched_sizes=True,
            )
        else:
            self.bert = TFBertModel.from_pretrained("google-bert/bert-base-uncased")

        self.bert.trainable = False  # freeze pre-trained weights
        self.broadcaster = layers.Dense(units=self.projection_dim)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = self.tokenizer(input)
        x = self.bert(**x)
        x = tf.expand_dims(self.broadcaster(x.pooler_output), 1)
        return x

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "max_seq_length": self.max_seq_length,
            }
        )
        return config


class DistilBertEncoder(layers.Layer):
    """Encoder based on Google's DistilBERT model.

    Attributes:
        projection_dim (_type_): Length of final hidden state vector output by encoder.
        max_seq_length (_type_): Length to which tokenized sequences are trimmed.
        vocab_file (_type_): Location of vocab file in filesystem.
        tokenizer (_type_): choice of tokenizer.
        bert (_type_): Pretrained DistilBERT model.
        broadcaster (_type_): Dense layer that broadcasts output of pretrained BERT model into `projection_dim` dimensionality.
    """  # noqa: E501

    def __init__(
        self, projection_dim: int, vocab_file: str, max_seq_length: int = None
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.max_seq_length = max_seq_length
        self.vocab_file = vocab_file

        if self.max_seq_length:
            self.tokenizer = MyBertTokenizerTrimmed(
                self.max_seq_length, self.vocab_file
            )
            self.bert = TFDistilBertModel.from_pretrained(
                "distilbert/distilbert-base-uncased",
                max_position_embeddings=self.max_seq_length,
                ignore_mismatched_sizes=True,
            )
        else:
            self.tokenizer = MyBertTokenizer()
            self.bert = TFDistilBertModel.from_pretrained(
                "distilbert/distilbert-base-uncased"
            )

        self.bert.trainable = False  # freeze pre-trained weights
        self.broadcaster = layers.Dense(units=self.projection_dim)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """Forward propagation."""
        x = self.tokenizer(input)
        x.pop("token_type_ids")
        x = self.bert(**x)
        x = self.broadcaster(x.last_hidden_state)
        return x

    def get_config(self) -> dict[str,]:
        """Updates config object to enable model to be loaded from disk."""
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "vocab_file": self.vocab_file,
                "max_seq_length": self.max_seq_length,
            }
        )
        return config
