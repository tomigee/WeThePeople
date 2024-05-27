# AWS: Imports
import os
import argparse
import json
import logging
import sys
from time import sleep

# Imports
import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow import distribute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses

from src.modeling.preprocess.tf_pipeline import build
from modeling.models.custom_models import MyCustomModel1


# AWS: Parse params
def parse_args():
    # example command: python ML.py --model_dir "artifacts" --train_data_dir "datasets/test_training data" --output_data_dir "artifacts"  # noqa: E501

    parser = argparse.ArgumentParser()

    # Hyperparameters and algorithm parameters are described here
    parser.add_argument("--fred_series_id", type=str, default="GDPC1")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--batch_size_per_worker", type=int, default=1)
    parser.add_argument("--n_vocab", type=int, default=100000)
    parser.add_argument("--label_seq_length", type=int, default=20)
    parser.add_argument("--series_seq_length", type=int, default=40)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--stack_height", type=int, default=1)
    parser.add_argument("--d_values", type=int, default=12)
    parser.add_argument("--d_keys", type=int, default=12)
    parser.add_argument("--encoder_max_seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--distributed", type=str, default=None)

    # Location where trained model will be stored. Default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    # Location of input training data
    parser.add_argument(
        "--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
    )
    # Location where model artifacts will be stored. Default set by SageMaker, /opt/ml/output/data
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )

    args = parser.parse_args()
    return args


def setup(distributed=None):
    if distributed:
        # Distributed training stuff
        if distributed == "tf":

            def set_tf_config(max_retries=3):
                os.environ.pop("TF_CONFIG", None)
                retries = 0
                while retries < max_retries:
                    try:
                        with open(
                            "/opt/ml/input/config/resourceconfig.json", "r"
                        ) as file:
                            node_config = json.load(file)
                            break
                    except Exception:
                        retries += 1
                        sleep(5)

                list_of_workers = node_config["hosts"]
                port = "12345"  # hard-coded
                for i in range(len(list_of_workers)):
                    list_of_workers[i] = list_of_workers[i] + ":" + port
                current_worker = node_config["current_host"]
                tf_config = {
                    "cluster": {"worker": list_of_workers},
                    "task": {
                        "type": "worker",
                        "index": list_of_workers.index(current_worker + ":" + port),
                    },
                }
                os.environ["TF_CONFIG"] = json.dumps(tf_config)
                return tf_config

            # TODO: Move to requirements.txt
            os.system("pip install tf-nightly")
            tf_config = set_tf_config()
            num_workers = len(tf_config["cluster"]["worker"])
            print(f"Number of workers: {num_workers}")
            return num_workers

        elif distributed == "horovod":
            hvd.init()
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[hvd.local_rank()], "GPU"
                )
            return hvd.size()

    return 1


def script_to_run(args, num_workers):

    # Setup save directories
    model_dir = os.environ.get("SM_MODEL_DIR")
    logs_location = f"{args.output_data_dir}/logs"
    if not os.path.exists(logs_location):
        os.makedirs(logs_location)
    # is there an environment variable for this?
    checkpoint_dir = "/opt/ml/checkpoints"
    checkpoint_filename = "{epoch:02d}-{loss:.2f}.weights.h5"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Setup logger (for AWS tuning)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    def execute_training_routine(args):
        def initialize_model(model_hparams):
            return MyCustomModel1(**model_hparams)

        def get_training_options(train_hparams, val_data):

            config = {}
            config["optimizer"] = Adam(learning_rate=train_hparams["learning_rate"])
            config["loss"] = losses.MeanSquaredError()
            config["metrics"] = ["mse", "mape", "mae"]
            config["verbose"] = 2
            config["steps_per_epoch"] = None
            config["epochs"] = train_hparams["epochs"]
            config["callbacks"] = None
            config["validation_data"] = val_data
            return config

        def compile_model(model, opts, distributed):
            compile_keys = ["optimizer", "loss", "metrics"]
            compile_opts = {key: opts[key] for key in compile_keys}
            if distributed == "tf":
                pass
            elif distributed == "horovod":
                compile_opts["experimental_run_tf_function"] = False
                compile_opts["optimizer"] = hvd.DistributedOptimizer(
                    compile_opts["optimizer"],
                    backward_passes_per_step=1,
                    average_aggregated_gradients=True,
                )
            else:
                pass

            model.compile(**compile_opts)

        def fit_model(train_data, model, opts, train_hparams, steps_per_epoch):
            fit_keys = [
                "verbose",
                "epochs",
                "steps_per_epoch",
                "callbacks",
                "validation_data",
            ]
            fit_opts = {key: opts[key] for key in fit_keys}
            if train_hparams["distributed"] == "tf":
                pass  # populate when you figure out tf distributed
            elif train_hparams["distributed"] == "horovod":
                fit_opts["steps_per_epoch"] = steps_per_epoch  # noqa: E501
                fit_opts["callbacks"] = [
                    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                    hvd.callbacks.MetricAverageCallback(),
                    hvd.callbacks.LearningRateWarmupCallback(
                        initial_lr=train_hparams["learning_rate"],
                        warmup_epochs=3,
                        verbose=1,
                    ),
                ]
                if hvd.rank() == 0:
                    fit_opts["callbacks"].append(
                        tf.keras.callbacks.ModelCheckpoint(
                            os.path.join(checkpoint_dir, checkpoint_filename),
                            save_best_only=True,
                            save_weights_only=True,
                        )
                    )
                    fit_opts["callbacks"].append(
                        tf.keras.callbacks.TensorBoard(
                            log_dir=train_hparams["logs_location"]
                        )
                    )
                    fit_opts["verbose"] = 2
                else:
                    fit_opts["verbose"] = 0
            else:
                fit_opts["callbacks"] = [
                    tf.keras.callbacks.TensorBoard(
                        log_dir=train_hparams["logs_location"]
                    )
                ]

            history = model.fit(train_data, **fit_opts)
            return history

        def log_and_save_data(model, test_data, train_hparams, training_history):
            def log_metrics():
                metrics_data = {}
                # Log training and validation metrics
                for key in training_history.history:
                    logger.info(f"{key}: {training_history.history[key][-1]}")
                    metrics_data[key] = float(training_history.history[key][-1])

                # Log test metrics
                for i in range(len(model.metrics_names)):
                    logger.info(
                        f"test_{model.metrics_names[i]}: {evaluation_results[i]}"
                    )
                    metrics_data[f"test_{model.metrics_names[i]}"] = float(
                        evaluation_results[i]
                    )
                return metrics_data

            def save_data(metrics_data):
                # Save artifacts
                with open(train_hparams["metrics_location"], "w") as f:
                    json.dump(metrics_data, f)
                model_location = train_hparams["model_dir"] + "/model"
                model.save(model_location, save_format="tf")

            evaluation_results = model.evaluate(test_data)
            if train_hparams["distributed"] == "tf":
                # Change this code once you figure out tf.distribute
                metrics_data = log_metrics()
                save_data(metrics_data)
            elif train_hparams["distributed"] == "horovod":
                if hvd.rank() == 0:
                    metrics_data = log_metrics()
                    save_data(metrics_data)
            else:
                metrics_data = log_metrics()
                save_data(metrics_data)

        model_hparams = {
            "decoder_stack_height": args.stack_height,
            "d_values": args.d_values,
            "d_keys": args.d_keys,
            "h_model": args.num_heads,
            "decoder_dropout_rate": args.dropout_rate,
            "n_decoder_vocab": args.n_vocab,
            "label_seq_length": args.label_seq_length,
            "encoder_max_seq_len": args.encoder_max_seq_len,
            "vocab_file": args.train_data_dir + "/Bert_Vocabulary.txt",
        }

        pipeline_hparams = {
            "training_data_folder": args.train_data_dir,
            "fred_series_id": args.fred_series_id,
            "series_seq_length": args.series_seq_length,
            "label_seq_length": args.label_seq_length,
            "n_vocab": args.n_vocab,
            "num_threads": args.num_threads,
            "local_batch_size": args.batch_size_per_worker,
            "distributed": args.distributed,
        }

        train_hparams = {
            "learning_rate": args.learning_rate * num_workers,
            "epochs": args.epochs,
            "distributed": args.distributed,
            "model_dir": model_dir,
            "logs_location": logs_location,
            "metrics_location": f"{args.output_data_dir}/metrics.json",
        }

        train_data, val_data, test_data, steps_per_epoch = build(**pipeline_hparams)

        if train_hparams["distributed"] == "tf":
            strategy = distribute.MultiWorkerMirroredStrategy()
            with strategy.scope():
                my_model = initialize_model(model_hparams)
                opts = get_training_options(train_hparams, val_data)
                compile_model(my_model, opts, train_hparams["distributed"])
            training_history = fit_model(
                train_data, my_model, opts, train_hparams, steps_per_epoch
            )
            log_and_save_data(my_model, test_data, train_hparams, training_history)
        else:
            my_model = initialize_model(model_hparams)
            opts = get_training_options(train_hparams, val_data)
            compile_model(my_model, opts, train_hparams["distributed"])
            training_history = fit_model(
                train_data, my_model, opts, train_hparams, steps_per_epoch
            )
            log_and_save_data(my_model, test_data, train_hparams, training_history)

    execute_training_routine(args)


if __name__ == "__main__":
    args = parse_args()
    num_workers = setup(args.distributed)
    script_to_run(args, num_workers)
