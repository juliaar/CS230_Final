''' To call (Train the model): '''
#python train_model.py --data_dir "D:\CS230-Datasets\EgoGesture\\64x64_gestures" --model_dir "C:\Users\Julia Arnardottir\PycharmProjects\CS230_Final\experiments\\base_model"
#tensorboard --logdir="C:\Users\Julia Arnardottir\PycharmProjects\CS230_Final\experiments\\base_model\train_summaries" --port 6006
#tensorboard --logdir="C:\Users\Julia Arnardottir\PycharmProjects\CS230_Final\experiments\base_model\train_summaries" --port 6006


import argparse
import logging
import os
import tensorflow as tf

from model.input_data import input_def
from model.model_setup import model_def
from model.training_defs import train_and_evaluate
from model.utilities import Params
from model.utilities import set_logger
from model.utilities import splitter


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='C:\\Users\\Julia Arnardottir\\PycharmProjects\\CS230_Final\experiments\\base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='D:\\CS230-Datasets\\EgoGesture\\64x64_gestures',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwriting
    '''
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"
    '''
    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train_gestures")
    dev_data_dir = os.path.join(data_dir, "dev_gestures")

    # Get the filenames from the train and dev sets

    train_filenames = []
    train_labels = []
    train_dirs = [x[0] for x in tf.gfile.Walk(train_data_dir)]
    count = 0
    for train_dir in train_dirs:
        # We only want the subfolders:
        if train_dir in train_data_dir:
            continue
        train_filenames.append([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')])
        train_labels.append(int(train_filenames[count][1].split(splitter)[-1][:2]))
        count += 1

    eval_filenames = []
    eval_labels = []
    eval_dirs = [x[0] for x in tf.gfile.Walk(dev_data_dir)]
    count = 0
    for eval_dir in eval_dirs:
        # We only want the subfolders:
        if eval_dir in dev_data_dir:
            continue
        eval_filenames.append([os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith('.jpg')])
        eval_labels.append(int(eval_filenames[count][1].split(splitter)[-1][:2]))
        count += 1

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    # Create the two iterators over the two datasets
    train_inputs = input_def(True, train_filenames, train_labels, params)
    eval_inputs = input_def(False, eval_filenames, eval_labels, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_def('train', train_inputs, params)
    eval_model_spec = model_def('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
