import os

import tensorflow  as tf


class Dataset(object):
    """A simple class for handling data sets."""

    def __init__(self, name, subset, data_dir):
        """
        Initializes dataset using a subset and the path to the data.

        :param name:        [string]  Name of the data set.
        :param subset:      [string]  Name of the subset.
        :param data_dir:    [string]  Path to the data directory.
        """
        assert subset in self.available_subsets(), self.available_subsets()
        self._name = name
        self._subset = subset
        self._data_dir = data_dir

    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        pass

    def download_message(self):
        """Prints a download message for the Dataset."""
        pass

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    def data_files(self):
        """
        Returns a python list of all (sharded) data subset files.

        :return             [list]  python list of all (sharded) data set files.
        """
        tf_record_pattern = os.path.join(self.data_dir, '%s-*' % self.subset)
        data_files_list = tf.gfile.Glob(tf_record_pattern)
        if not data_files_list:
            log.error('No files found for dataset {}/{} at {}'.format(self.name, self.subset,
                                                                      self.data_dir))
            self.download_message()
            exit(-1)
        return data_files_list

    def reader(self):
        """
        Returns reader for a single entry from the data set.

        :return            [object]  Reader object that reads the data set.
        """
        return tf.TFRecordReader()

    @property
    def name(self):
        """Returns the name of the dataset."""
        return self._name

    @property
    def subset(self):
        """Returns the name of the subset."""
        return self._subset

    @property
    def data_dir(self):
        """Returns the directory path of the dataset."""
        return self._data_dir


class CifarDataset(Dataset):
    """CIFAR data set."""

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation', 'trainval', 'test']

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train':
            return 45000
        if self.subset == 'validation':
            return 5000
        if self.subset == 'trainval':
            return 50000
        if self.subset == 'test':
            return 10000

    def download_message(self):
        pass


class Cifar10Dataset(CifarDataset):
    """CIFAR-10 data set."""

    def __init__(self, subset, data_dir):
        super(Cifar10Dataset, self).__init__('CIFAR-10', subset, data_dir)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 10


class Cifar100Dataset(CifarDataset):
    """CIFAR-100 data set."""

    def __init__(self, subset, data_dir):
        super(Cifar100Dataset, self).__init__('CIFAR-100', subset, data_dir)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 100


class NoisyCifarDataset(Dataset):
    """CIFAR data set."""

    def __init__(self, name, subset, data_dir, num_clean, num_val):
        super(NoisyCifarDataset, self).__init__(name, subset, data_dir)
        self._num_clean = num_clean
        self._num_val = num_val

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train_noisy':
            return 50000 - self.num_val - self.num_clean
        elif self.subset == 'train_clean':
            return self.num_clean
        elif self.subset in ['validation', 'validation_noisy']:
            return self.num_val
        elif self.subset == 'test':
            return 10000

    def download_message(self):
        pass

    def available_subsets(self):
        return ['train_noisy', 'train_clean', 'validation', 'validation_noisy', 'test']

    @property
    def num_clean(self):
        return self._num_clean

    @property
    def num_val(self):
        return self._num_val


class NoisyCifar10Dataset(NoisyCifarDataset):
    """CIFAR-10 data set."""

    def __init__(self, subset, data_dir, num_clean, num_val):
        super(NoisyCifar10Dataset, self).__init__('Noisy CIFAR-10', subset, data_dir, num_clean,
                                                  num_val)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 10


class NoisyCifar100Dataset(NoisyCifarDataset):
    """CIFAR-100 data set."""

    def __init__(self, subset, data_dir, num_clean, num_val):
        super(NoisyCifar100Dataset, self).__init__('Noisy CIFAR-100', subset, data_dir, num_clean,
                                                   num_val)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 100
