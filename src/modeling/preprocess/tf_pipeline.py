import os
import tensorflow as tf
from tensorflow.keras import layers
import horovod.tensorflow.keras as hvd

import numpy as np
import re


def calculate_steps_per_epoch(
    n_total_examples, n_workers, test_split, local_batch_size
) -> int:
    """Calculate `steps_per_epoch` parameter when training with Tensorflow in a distributed environment.

    Args:
        n_total_examples (int): Total number of examples available to train on
        n_workers (int): Number of workers available
        test_split (int): Percentage of dataset to be used as test data.
        local_batch_size (int): Batch size of each worker

    Returns:
        The appropriate `steps_per_epoch` for the given configuration
    """  # noqa: E501
    min_shard_size = n_total_examples // n_workers
    divisor = round(100 / test_split)
    n_test_examples = -(-min_shard_size // divisor)  # ceiling division
    n_train_examples = min_shard_size - n_test_examples
    steps_per_epoch = n_train_examples // local_batch_size
    return steps_per_epoch


def get_training_data(
    path: str, fred_series_id: str, return_filepaths: bool
) -> list[str | None]:
    """Get filesystem location of training data files (bill text, fred series), or the contents of the files themselves.

    Args:
        path (str): Filesystem location of folder containing all training data.
        fred_series_id (str): Identifier of FRED series of interest.
        return_filepaths (bool): If True, return the filesystem locations of each file containing training data.

    Returns:
        The contents of training data files, or the filesystem locations of those files.
    """  # noqa: E501
    data_folder = os.path.basename(path)
    fred_series_folder = os.path.join(path, fred_series_id)  # untested
    pattern = r".+_(.+)"
    matches = re.match(pattern, data_folder)
    prefix = matches.group(1)
    x1_filepath = os.path.join(path, prefix + ".txt")
    if os.path.exists(fred_series_folder):  # untested
        x2_filepath = os.path.join(fred_series_folder, fred_series_id + "_series.csv")
        y_filepath = os.path.join(fred_series_folder, fred_series_id + "_label.csv")
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


def get_train_val_split(
    val_split: int | float, dataset: tf.data.Dataset, **kwargs: dict[str,]
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Split a dataset into training dataset and validation dataset.

    Args:
        val_split (int | float): Percentage of dataset to allocate to the validation dataset.
        dataset (tf.data.Dataset): The dataset to be split.

    Returns:
        The training dataset and validation dataset.
    """
    default_batch_size = 32
    default_num_parallel_calls = 8

    temp = dataset.shuffle(1000, reshuffle_each_iteration=False)
    divisor = round(100 / val_split)

    def is_val(x, y):
        return x % divisor == 0

    def is_train(x, y):
        return not is_val(x, y)

    def recover(x, y):
        return y

    val_dataset = temp.enumerate().filter(is_val).map(recover)
    if kwargs["distributed"] is not None:
        train_dataset = temp.enumerate().filter(is_train).map(recover).repeat()
    else:
        train_dataset = temp.enumerate().filter(is_train).map(recover)

    if "batch_size" in kwargs and "num_parallel_calls" in kwargs:
        val_dataset = val_dataset.batch(
            batch_size=kwargs["batch_size"],
            num_parallel_calls=kwargs["num_parallel_calls"],
        )
        train_dataset = train_dataset.batch(
            batch_size=kwargs["batch_size"],
            num_parallel_calls=kwargs["num_parallel_calls"],
        )
    elif "batch_size" in kwargs:
        val_dataset = val_dataset.batch(
            batch_size=kwargs["batch_size"],
            num_parallel_calls=default_num_parallel_calls,
        )
        train_dataset = train_dataset.batch(
            batch_size=kwargs["batch_size"],
            num_parallel_calls=default_num_parallel_calls,
        )
    else:
        val_dataset = val_dataset.batch(
            batch_size=default_batch_size, num_parallel_calls=default_num_parallel_calls
        )
        train_dataset = train_dataset.batch(
            batch_size=default_batch_size, num_parallel_calls=default_num_parallel_calls
        )

    return train_dataset, val_dataset


def get_train_test_split(
    test_split: int | float, data_paths: list[str]
) -> tuple[list[str]]:
    """Split a list of training data paths into training and testing sets.

    Args:
        test_split (int | float): Percentage of examples to allocate to test set.
        data (list[str]): List of filesystem paths with each entry representing a training example.

    Returns:
        Train dataset paths and test dataset paths.
    """
    divisor = round(100 / test_split)

    def is_test(x):
        return x[0] % divisor == 0

    def is_train(x):
        return not is_test(x)

    test_data = [item for idx, item in filter(is_test, enumerate(data_paths))]
    train_data = [item for idx, item in filter(is_train, enumerate(data_paths))]

    return train_data, test_data


# This is where we pull out the actual data we want
def load_data(path: str, is_csv: bool = False) -> tf.Tensor:
    """Read out the actual contents of training example files.

    Args:
        path (str): Filesystem location of file to be read.
        is_csv (bool, optional): File is a csv. Defaults to False.

    Returns:
        Contents of the file as a tensor.
    """
    if is_csv:
        decoded_path = bytes.decode(path.numpy())
        x = np.loadtxt(decoded_path, dtype=str, delimiter=",", skiprows=1)
        relevant_series = (
            x[:, 3].astype(float).round().astype("int32")
        )  # reduce to int for homogeneity across fred series
        x = tf.constant(relevant_series)
    else:
        x = tf.io.read_file(path)
    return x


def set_shape(item: tf.Tensor, shape: list[int]) -> tf.Tensor:
    """Set the shape of a tensor

    Args:
        item (tf.Tensor): Tensor whose shape will be set.
        shape (list[int]): Desired shape.

    Raises:
        ValueError: If `shape` is not a list.

    Returns:
        Same tensor as `item`, but with the shape attribute set.
    """
    if not isinstance(shape, list):
        raise ValueError("shape must be a List")
    item.set_shape(shape)
    return item


def build(
    training_data_folder: str,
    fred_series_id: str,
    series_seq_length: int,
    label_seq_length: int,
    n_vocab: int,
    num_threads: int,
    local_batch_size: int,
    distributed: str | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:
    """Build the train, validation, test datasets (tf.data.Dataset) from the downloaded dataset on disk, and additionally calculate the `steps_per_epoch`

    Args:
        training_data_folder (str): Filesystem location of folder containing all training data.
        fred_series_id (str): Identifier of FRED series of interest.
        series_seq_length (int): Length of FRED time series used as a feature.
        label_seq_length (int): Length of FRED time series used as a target.
        n_vocab (int): Number of entries in (numerical) vocabulary used for encoding FRED time series.
        num_threads (int): Number of threads to use to load the data.
        local_batch_size (int): Batch size of each worker in a distributed training environment.
        distributed (str | None, optional): Service used to distribute the training job. Defaults to None.

    Returns:
        Train dataset, validation dataset, test dataset, and `steps_per_epoch`.
    """  # noqa: E501

    batch_size = local_batch_size
    test_split = 20
    val_split = 10
    # Get folders only
    training_example_folders = []
    for folder in os.listdir(training_data_folder):
        path = os.path.join(training_data_folder, folder)
        if os.path.isdir(path):
            training_example_folders.append(path)

    if distributed:
        training_example_folders.sort()  # for repeatability across shards

    # Separate training from testing data
    training_example_folders, test_example_folders = get_train_test_split(
        test_split, training_example_folders
    )
    example_folders = [training_example_folders, test_example_folders]

    # FOR TRAIN AND TEST
    data_filepaths = [None, None]  # initialize list
    for i in range(len(example_folders)):
        # Get feature and label filepaths from folders
        data_filepaths[i] = {"billtext": [], "series": [], "label": []}
        for training_example in example_folders[i]:
            x1, x2, y = get_training_data(
                training_example, fred_series_id, return_filepaths=True
            )
            if (x2 is not None) and (y is not None):
                data_filepaths[i]["billtext"].append(x1)
                data_filepaths[i]["series"].append(x2)
                data_filepaths[i]["label"].append(y)

    # FOR TRAIN ONLY
    n_total_examples = len(data_filepaths[0]["label"])

    # FOR TRAIN AND TEST
    training_data = [None, None]  # initialize list
    for i in range(len(data_filepaths)):
        # Hark! A pipeline!
        for key in data_filepaths[i]:
            data_filepaths[i][key] = tf.data.Dataset.from_tensor_slices(
                data_filepaths[i][key]
            )

            if i == 0 and distributed == "horovod":  # don't shard test data
                data_filepaths[i][key] = data_filepaths[i][key].shard(
                    hvd.size(), hvd.rank()
                )

        # Do dataset mappings
        my_tokenizer = layers.IntegerLookup(
            vocabulary=np.arange(n_vocab, dtype="int32")
        )
        seq_length = {"series": series_seq_length, "label": label_seq_length}
        data = {}
        for key in data_filepaths[i]:
            if key == "billtext":
                # Process billtext data
                data[key] = (
                    data_filepaths[i][key]
                    .map(
                        lambda x: tf.py_function(load_data, [x, False], tf.string),
                        num_parallel_calls=num_threads,
                    )
                    .map(
                        lambda item: set_shape(item, []), num_parallel_calls=num_threads
                    )
                )
            else:
                # Process label and series data
                data[key] = (
                    data_filepaths[i][key]
                    .map(
                        lambda x: tf.py_function(load_data, [x, True], tf.int32),
                        num_parallel_calls=num_threads,
                    )
                    .map(
                        lambda item: set_shape(item, [seq_length[key]]),
                        num_parallel_calls=num_threads,
                    )
                    .map(my_tokenizer.call, num_parallel_calls=num_threads)
                )

        # Zip
        training_data[i] = tf.data.Dataset.zip(
            (data["billtext"], data["series"]), data["label"]
        )

    # Custom logic for distributed training
    if distributed == "tf":
        # Recalculate batch size
        # batch_size = local_batch_size * num_workers
        pass
    if distributed == "horovod":
        steps_per_epoch = calculate_steps_per_epoch(
            n_total_examples, hvd.size(), val_split, batch_size
        )
    else:
        steps_per_epoch = None

    # FOR TRAIN ONLY
    train_data, val_data = get_train_val_split(
        val_split,
        training_data[0],
        batch_size=batch_size,
        num_parallel_calls=num_threads,
        distributed=distributed,
    )
    test_data = training_data[1].batch(
        batch_size=batch_size, num_parallel_calls=num_threads
    )
    return train_data, val_data, test_data, steps_per_epoch
