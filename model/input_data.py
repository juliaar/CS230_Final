import tensorflow as tf
import numpy as np

def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    #filename = tf.reshape(filename, [5, 1])

    #[n, _] = filename.shape
    #print(label[0].shape)

    for k in range(len(filename)):

        image_string = tf.read_file(filename[k])

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)

        if k == 0:
            resized_image = tf.image.resize_images(image, [size, size])
        else:
            re_image_k = tf.image.resize_images(image, [size, size])
            resized_image = tf.concat([resized_image, re_image_k], axis=2)

    # outputs resizes image with shape (64,64,15) & its label
    return resized_image, label

# params.train_size = len(train_filenames)/5
# train_inputs = input_def(True, train_filenames, train_labels, params)
# eval_inputs = input_def(False, eval_filenames, eval_labels, params)
def input_def(is_training, filenames, labels, params):
    """Input function for the EgoGesture dataset.

    The filenames have format "{class label}_{part number (0-4)}_{total example number}.jpg".
    For instance: "data_dir/23_2_806.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images (5 images together)
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    #train_fn = lambda f, l: _parse_function(f, l, params.image_size)
    images = []
    for i in range(len(filenames)):
        im_i, _ = _parse_function(filenames[i], labels[i], params.image_size)
        images.append(im_i)

    num_samples = len(filenames)
    if is_training:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
        dataset = dataset.batch(params.batch_size).prefetch(1)
        #dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
        #    .map(train_fn, num_parallel_calls=params.num_parallel_calls)
        #    .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
        #    .batch(params.batch_size)
        #    .prefetch(1)  # make sure you always have one batch ready to serve
        #)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
        dataset = dataset.batch(params.batch_size).prefetch(1)
        #dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
        #    .map(train_fn)
        #    .batch(params.batch_size)
        #    .prefetch(1)  # make sure you always have one batch ready to serve
        #)

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
