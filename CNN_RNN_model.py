# python CNN_RNN_model.py --data_dir "D:\CS230-Datasets\EgoGesture\\64x64_gestures" --model_dir "C:\Users\Julia Arnardottir\PycharmProjects\CS230_Final\experiments\\lrcn_model"
# tensorboard --logdir="C:\Users\Julia Arnardottir\PycharmProjects\CS230_Final\lrcn_data\logs"

import os.path
import argparse
import logging
import tensorflow as tf
from model.utilities import Params
from model.utilities import set_logger
from model.utilities import splitter
from model.LSTM_model import train_LRCN_model


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='C:\\Users\\Julia Arnardottir\\PycharmProjects\\CS230_Final\experiments\\lrcn_model',
                    help="Experiment directory containing params_lrcn.json")
parser.add_argument('--data_dir', default='D:\\CS230-Datasets\\EgoGesture\\64x64_gestures',
                    help="Directory containing the dataset")

def get_data_info(args):

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train_lrcn.log'))

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
        if eval_dir in dev_data_dir:
            continue
        eval_filenames.append([os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith('.jpg')])
        eval_labels.append(int(eval_filenames[count][1].split(splitter)[-1][:2]))
        count += 1

    return train_filenames, train_labels, eval_filenames, eval_labels


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params_lrcn.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    train_filenames, train_labels, eval_filenames, eval_labels = get_data_info(args)

    logging.info("Creating and evaluating the CNN-LSTM model...")
    train_LRCN_model(params, train_filenames, train_labels, eval_filenames, eval_labels)


