''' To call: '''
#python define_dataset.py --data_dir "D:\CS230-Datasets\EgoGesture" --output_dir "D:\CS230-Datasets\EgoGesture\64x64_gestures"


import argparse
import os
from model.ourdata_defs import get_data_and_save

SIZE = 64
gestures = [23, 52, 53]
percent_dev = 10
percent_test = 10

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='D:\\CS230-Datasets\\EgoGesture', help="Directory with the EgoGesture dataset")
parser.add_argument('--output_dir', default='D:\\CS230-Datasets\\EgoGesture\\64x64_gestures', help="Where to write the new data")


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    get_data_and_save(args.data_dir, args.output_dir, gestures, percent_dev, percent_test, SIZE)

    print("Done building dataset")
