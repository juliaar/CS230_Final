
import argparse
import random
import os

from ourdata_defs import get_data

SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='D:\\CS230-Datasets\\EgoGesture', help="Directory with the EgoGesture dataset")
parser.add_argument('--output_dir', default='D:\\CS230-Datasets\\EgoGesture\\64x64_gestures', help="Where to write the new data")

gestures = [23, 52, 53]
percent_dev = 10
percent_test = 10

#TODO: update this file and combine with 'dataDefs.py' to fit our dataset

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))




    print("Done building dataset")
