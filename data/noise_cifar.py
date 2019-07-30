import os

import numpy as np
import tensorflow as tf


class NoiseCifar(object):

    def __init__(self, root, method="train", noisy_ratio=0.4, batch_size=100):
        self.root = root
        self.data = np.load(os.path.join(self.root, "noise_cifar_{}.npz".format(noisy_ratio)))
        self.method = method
        if method == "train":
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.data['train_image'], self.data['train_label'])).repeat().shuffle(10 * batch_size)
        elif method == "meta":
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.data['meta_image'], self.data['meta_label'])).repeat().shuffle(10 * batch_size)
        elif method == "val":
            dataset = tf.data.Dataset.from_tensor_slices((self.data['val_image'], self.data['val_label']))
        elif method == "test":
            dataset = tf.data.Dataset.from_tensor_slices((self.data['test_image'], self.data['test_label']))
        else:
            raise BaseException()

        dataset = dataset.batch(batch_size)
        self.iteration = dataset.make_one_shot_iterator()

    def __iter__(self):
        return self

    def __next__(self):
        return self.iteration.get_next()
