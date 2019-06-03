# python start_run.py --model_dir "C:\Users\Julia Arnardottir\PycharmProjects\VisionExample\experiments\base_model"

import cv2
import os
import argparse
import logging
import tensorflow as tf
from PIL import Image
from model.utilities import Params
from model.utilities import set_logger
from classification_defs import sample_def

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Directory containing params.json")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")

zeros = 5


def getFrame(count, zeros):
  size = 64
  hasFrames, image = vidcap.read()
  name = "frame" + str(count).zfill(zeros) + ".jpg"
  if hasFrames:
    cv2.imwrite(name, image)  # save frame as JPG file
  smallimage = Image.open(name)
  smallimage = smallimage.resize((size, size), Image.BILINEAR)
  smallimage.save(name)
  return hasFrames

def remove_img(count, zeros):
    number = count - 5
    img_name = "frame" + str(number).zfill(zeros) + ".jpg"
    os.remove(img_name)
    return True


if __name__ == '__main__':

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'classify.log'))

    logging.info("Deleting old images...")
    files = [f for f in os.listdir('.') if f.endswith('.jpg')]
    for file in files:
        os.remove(file)

    logging.info("Opening webcam...")
    vidcap = cv2.VideoCapture(0)
    # automatically captures 30 fps (we want to show those but only save 3 fps)
    # print(vidcap.get(cv2.CAP_PROP_FPS))

    totcount = 0
    count = 0

    logging.info("Getting the first 5 frames...")
    while count < 5:
        # Capture frame-by-frame
        ret, frame = vidcap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)

        if (totcount % 10) == 0:
          count = count + 1
          # Save frame
          getFrame(count, zeros)

        totcount = totcount + 1

        # Stop if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logging.info("Starting classification...")
    # continue, and overwrite the 5 images one at a time
    while True:
        # Capture frame-by-frame
        ret, frame = vidcap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)

        if (totcount % 10) == 0:
            files = [f for f in os.listdir('.') if f.endswith('.jpg')]
            print(files)
            # Stack and run through the model to get prediction:
            prediction = sample_def(files, params)
            r = tf.rank(prediction)
            print(prediction)
            #sess = tf.Session()
            #print(sess.run(prediction))
            # Prepare for the next run through
            count = count + 1
            # Only want to store 5 images
            remove_img(count, zeros)
            # Save frame
            getFrame(count, zeros)

        # Don't want to run for too long while testing, delete when ready
        if totcount == 200:
          break

        totcount = totcount + 1

        # Stop if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    vidcap.release()
    cv2.destroyAllWindows()

