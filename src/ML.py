# AWS: Imports
import os
import argparse
import json
import joblib
import logging
import sys

# Imports
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses

from modeling.preprocess.get_training_data import compile_training_data
from modeling.models.custom_models import MyCustomModel1


# AWS: Parse params
def parse_args():
    # example command (create "artifacts" folder before this): python ML.py --model_dir "artifacts" --train_data_dir "datasets/test_training data" --output_data_dir "artifacts"

    parser = argparse.ArgumentParser()

    # Hyperparameters and algorithm parameters are described here
    parser.add_argument("--fred_series_id", type=str, default="GDPC1")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
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

    # Location where trained model will be stored. Default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    # Location of input training data
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    # Location where model artifacts will be stored. Default set by SageMaker, /opt/ml/output/data
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))

    args = parser.parse_args()
    return args


def setup():
    # AWS: Virtual Env & Pip installs
    os.system("python -m venv .venv")
    os.system("source .venv/bin/activate")
    os.system("pip install --upgrade pip")
    os.system("pip install -r requirements.txt")


def script_to_run(args):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # Set input pipeline hyperparameters - User customizable
    pipeline_hparams = {
        "training_data_folder": args.train_data_dir,
        "fred_series_id": args.fred_series_id,
        "series_seq_length": args.series_seq_length,
        "label_seq_length": args.label_seq_length,
        "n_vocab": args.n_vocab,
        "num_threads": args.num_threads,
        "batch_size": args.batch_size
    }

    # Set model hyperparameters - User customizable
    model_hparams = {
        "decoder_stack_height": args.stack_height,
        "d_values": args.d_values,
        "d_keys": args.d_keys,
        "h_model": args.num_heads,
        "decoder_dropout_rate": args.dropout_rate,
        "n_decoder_vocab": args.n_vocab,
        "label_seq_length": args.label_seq_length,
        "encoder_max_seq_len": args.encoder_max_seq_len
    }

    # Set training hyperparameters - User customizable
    train_hparams = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs
    }

    train_data, test_data = compile_training_data(**pipeline_hparams)

    my_model = MyCustomModel1(**model_hparams)
    my_model.compile(optimizer=Adam(learning_rate=train_hparams["learning_rate"]),
                     loss=losses.MeanSquaredError(),
                     metrics=['mse'])
    my_model.fit(train_data, epochs=train_hparams["epochs"], verbose=2)
    output = my_model.evaluate(test_data)
    metrics_data = {}
    for i in range(len(my_model.metrics_names)):
        logger.info(f"{my_model.metrics_names[i]}: {output[i]}")  # metric for the AWS tuner
        metrics_data[my_model.metrics_names[i]] = output[i]

    metrics_location = f"{args.output_data_dir}/metrics.json"
    model_location = f"{args.model_dir}/model"

    with open(metrics_location, "w") as f:
        json.dump(metrics_data, f)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(model_location, "wb") as f:
        joblib.dump(my_model, f)


if __name__ == "__main__":
    args = parse_args()
    setup()
    script_to_run(args)
