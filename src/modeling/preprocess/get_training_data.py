import os
import tensorflow as tf
from tensorflow.keras import layers
import horovod.tensorflow.keras as hvd

import numpy as np
import re


def calculate_steps_per_epoch(n_total_examples, n_workers, test_split, local_batch_size):
    min_shard_size = n_total_examples // n_workers
    divisor = round(100/test_split)
    n_test_examples = -(-min_shard_size//divisor)  # ceiling division
    n_train_examples = min_shard_size - n_test_examples
    steps_per_epoch = n_train_examples // local_batch_size
    return steps_per_epoch


def get_training_data(path, fred_series_id, return_filepaths):
    data_folder = os.path.basename(path)
    fred_series_folder = os.path.join(path, fred_series_id)  # untested
    pattern = r".+_(.+)"
    matches = re.match(pattern, data_folder)
    prefix = matches.group(1)
    x1_filepath = os.path.join(path, prefix + ".txt")
    if os.path.exists(fred_series_folder):  # untested
        x2_filepath = os.path.join(
            fred_series_folder, fred_series_id + "_series.csv")
        y_filepath = os.path.join(fred_series_folder,
                                  fred_series_id + "_label.csv")
    else:
        x2_filepath = None
        y_filepath = None
    paths = [x1_filepath, x2_filepath, y_filepath]

    if return_filepaths:
        return paths
    else:
        file_contents = []
        for path in paths:
            if path:
                file_contents.append(tf.io.read_file(path))
            else:
                file_contents.append(None)
        return file_contents


def get_train_test_split(test_split, dataset, **kwargs):
    default_batch_size = 32
    default_num_parallel_calls = 8

    temp = dataset.shuffle(1000, reshuffle_each_iteration=False)
    divisor = round(100/test_split)

    def is_test(x, y):
        return x % divisor == 0

    def is_train(x, y):
        return not is_test(x, y)

    def recover(x, y): return y

    test_dataset = temp.enumerate() \
        .filter(is_test) \
        .map(recover)
    train_dataset = temp.enumerate() \
                        .filter(is_train) \
                        .map(recover)

    if 'batch_size' in kwargs and 'num_parallel_calls' in kwargs:
        test_dataset = test_dataset.batch(
            batch_size=kwargs['batch_size'],
            num_parallel_calls=kwargs['num_parallel_calls']
        )
        train_dataset = train_dataset.batch(
            batch_size=kwargs['batch_size'],
            num_parallel_calls=kwargs['num_parallel_calls']
        )
    elif 'batch_size' in kwargs:
        test_dataset = test_dataset.batch(
            batch_size=kwargs['batch_size'],
            num_parallel_calls=default_num_parallel_calls
        )
        train_dataset = train_dataset.batch(
            batch_size=kwargs['batch_size'],
            num_parallel_calls=default_num_parallel_calls
        )
    else:
        test_dataset = test_dataset.batch(
            batch_size=default_batch_size,
            num_parallel_calls=default_num_parallel_calls
        )
        train_dataset = train_dataset.batch(
            batch_size=default_batch_size,
            num_parallel_calls=default_num_parallel_calls
        )

    return train_dataset, test_dataset


# This is where we pull out the actual data we want
def load_data(path, is_csv=False):
    if is_csv:
        decoded_path = bytes.decode(path.numpy())
        x = np.loadtxt(decoded_path, dtype=str, delimiter=",", skiprows=1)
        relevant_series = x[:, 3].astype(float).round().astype(
            'int32')  # reduce to int for homogeneity across fred series
        x = tf.constant(relevant_series)
    else:
        x = tf.io.read_file(path)
    return x


def set_shape(item, shape):
    if not isinstance(shape, list):
        raise ValueError("shape must be a List")
    item.set_shape(shape)
    return item


def compile_training_data(training_data_folder,
                          fred_series_id,
                          series_seq_length,
                          label_seq_length,
                          n_vocab,
                          num_threads,
                          local_batch_size,
                          distributed=None):

    batch_size = local_batch_size
    test_split = 20
    # Get folders that are valid training examples
    training_example_folders = []
    for folder in os.listdir(training_data_folder):
        path = os.path.join(training_data_folder, folder)
        if os.path.isdir(path):
            training_example_folders.append(path)

    if distributed:
        training_example_folders.sort()  # for repeatability across shards

    # Get feature and label filepaths from valid training example folders
    data_filepaths = {"billtext": [],
                      "series": [],
                      "label": []}
    for training_example in training_example_folders:
        x1, x2, y = get_training_data(
            training_example, fred_series_id, return_filepaths=True)
        if (x2 is not None) and (y is not None):
            data_filepaths["billtext"].append(x1)
            data_filepaths["series"].append(x2)
            data_filepaths["label"].append(y)
    n_total_examples = len(data_filepaths["label"])

    steps_per_epoch = calculate_steps_per_epoch(
        n_total_examples, hvd.size(), test_split, batch_size
    )

    # Hark! A pipeline!
    for key in data_filepaths:
        data_filepaths[key] = tf.data.Dataset.from_tensor_slices(
            data_filepaths[key])

        if distributed == "horovod":
            data_filepaths[key] = data_filepaths[key].shard(
                hvd.size(), hvd.rank())

    # Do dataset mappings
    my_tokenizer = layers.IntegerLookup(
        vocabulary=np.arange(n_vocab, dtype='int32'))
    seq_length = {"series": series_seq_length, "label": label_seq_length}
    data = {}
    for key in data_filepaths:
        if key == "billtext":
            # Process billtext data
            data[key] = data_filepaths[key] \
                .map(lambda x: tf.py_function(load_data, [x, False], tf.string),
                     num_parallel_calls=num_threads) \
                .map(lambda item: set_shape(item, []),
                     num_parallel_calls=num_threads)
        else:
            # Process label and series data
            data[key] = data_filepaths[key] \
                .map(lambda x: tf.py_function(load_data, [x, True], tf.int32),
                     num_parallel_calls=num_threads) \
                .map(lambda item: set_shape(item, [seq_length[key]]),
                     num_parallel_calls=num_threads) \
                .map(my_tokenizer.call, num_parallel_calls=num_threads)

    # Zip
    training_data = tf.data.Dataset.zip(
        (data["billtext"], data["series"]), data["label"])

    # Custom logic for distributed training
    if distributed == "tf":
        # Recalculate batch size
        # batch_size = local_batch_size * num_workers
        pass
    if distributed == "horovod":
        pass
    else:
        pass

    train_data, test_data = get_train_test_split(
        test_split,
        training_data,
        batch_size=batch_size, num_parallel_calls=num_threads
    )
    return train_data, test_data, steps_per_epoch
