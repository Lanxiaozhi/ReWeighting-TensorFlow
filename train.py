import os
from collections import namedtuple

import numpy as np
import tensorflow as tf
from config import train_config
from data.noise_cifar import NoiseCifar
from data.noise_cifar_dataset import NoisyCifar10Dataset
from data.noise_cifar_pipeline import NoisyCifarInputPipeline
from model.assign_weight_resnet import AssignWeightResNet
from tqdm import trange, tqdm


def get_data_inputs(data_dir, subset, is_training, batch_size, ):
    dataset = NoisyCifar10Dataset(num_clean=1000, num_val=5000, data_dir=data_dir, subset=subset)
    return NoisyCifarInputPipeline(dataset, is_training=is_training, batch_size=batch_size)


def _get_data_input(dataset, data_dir, split, bsize, is_training, seed, **kwargs):
    """Builds data input."""
    data = get_data_inputs(data_dir, split, is_training, bsize)
    batch = data.inputs(seed=seed)
    inp, label, idx, clean_flag = batch['image'], batch['label'], batch['index'], batch['clean']
    DataTuple = namedtuple('DataTuple', ['data', 'inputs', 'labels', 'index', 'clean_flag'])
    return DataTuple(data=data, inputs=inp, labels=label, index=idx, clean_flag=clean_flag)


def _get_data_inputs(bsize, seed=0):
    """Gets data input tensors."""
    # Compute the dataset directory for this experiment.
    data_name = FLAGS.dataset
    data_dir = os.path.join(FLAGS.data_root, data_name)
    print(data_dir)

    print('Building dataset')
    trn_data = _get_data_input(data_name, data_dir, 'train', bsize, True, seed)
    val_data = _get_data_input(data_name, data_dir, 'validation', bsize, False, seed)
    test_data = _get_data_input(data_name, data_dir, 'test', bsize, False, seed)

    class Datasets:
        train = trn_data
        val = val_data
        test = test_data

    return Datasets()


def get_noisy_data_inputs(bsize, seed=0):
    """Gets data input tensors."""
    # Compute the dataset directory for this experiment.
    data_name = FLAGS.dataset + '-noisy-clean{:d}-noise{:d}-val{:d}-seed{:d}'.format(
        FLAGS.num_clean, int(FLAGS.noise_ratio * 100), FLAGS.num_val, FLAGS.seed)
    data_dir = os.path.join(FLAGS.data_root, data_name)
    print(data_dir)

    # Generate TF records if not exist.
    # if not os.path.exists(data_dir):
    #     generate_noisy_cifar(FLAGS.dataset,
    #                          os.path.join(FLAGS.data_root, FLAGS.dataset), FLAGS.num_val,
    #                          FLAGS.noise_ratio, FLAGS.num_clean, data_dir, FLAGS.seed)

    print('Building dataset')
    dataset = FLAGS.dataset + '-noisy'
    trn_clean_data = _get_data_input(
        dataset,
        data_dir,
        'train_clean',
        bsize,
        True,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)
    trn_noisy_data = _get_data_input(
        dataset,
        data_dir,
        'train_noisy',
        bsize,
        True,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)
    val_data = _get_data_input(
        dataset,
        data_dir,
        'validation',
        bsize,
        False,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)
    val_noisy_data = _get_data_input(
        dataset,
        data_dir,
        'validation_noisy',
        bsize,
        False,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)
    test_data = _get_data_input(
        dataset,
        data_dir,
        'test',
        bsize,
        False,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)

    class Datasets:
        train_clean = trn_clean_data
        train_noisy = trn_noisy_data
        val = val_data
        val_noisy = val_noisy_data
        test = test_data

    return Datasets()


def get_data():
    # print('Config: {}'.format(MessageToString(config)))

    data_name = FLAGS.dataset + '-noisy-clean{:d}-noise{:d}-val{:d}-seed{:d}'.format(
        FLAGS.num_clean, int(FLAGS.noise_ratio * 100), FLAGS.num_val, FLAGS.seed)
    data_dir = os.path.join(FLAGS.data_root, data_name)

    # Initializes variables.


def main():
    batch_size = 100
    bsize_a = 100
    bsize_b = 100
    model_config = train_config['resnet_model_config']

    # iter_train = iter(NoiseCifar(root="/data/kouzhi/CIFAR_NOISE", method="train", batch_size=batch_size))
    # iter_meta = iter(NoiseCifar(root="/data/kouzhi/CIFAR_NOISE", method="meta", batch_size=batch_size))
    # train_image, train_label = next(iter_train)
    # meta_image, meta_label = next(iter_meta)

    with tf.variable_scope('Model'):
        model_main = AssignWeightResNet(config=model_config)

    ex_wts_a = tf.zeros([batch_size], dtype=tf.float32, name='ex_wts_a')
    with tf.variable_scope('Model', reuse=True):
        model_a = AssignWeightResNet(config=model_config, ex_wts=ex_wts_a)

    with tf.variable_scope('Model', reuse=True):
        wts_dict_new = model_a.wts_dict
        model_b = AssignWeightResNet(config=model_config, wts_dict=wts_dict_new)

    var_list_a = [model_a.wts_dict[k] for k in model_a.wts_dict.keys()]
    grads_a = tf.gradients(model_a.loss, var_list_a, gate_gradients=1)

    grads_b = tf.gradients(model_b.loss, var_list_a, gate_gradients=1)
    grads_ex_wts = tf.gradients(grads_a, [ex_wts_a], grads_b, gate_gradients=1)[0]

    ex_wts_plus = tf.maximum(grads_ex_wts, 0.0)
    ex_wts_sum = tf.reduce_sum(ex_wts_plus)
    ex_wts_sum += tf.cast(tf.equal(ex_wts_sum, 0.0), ex_wts_sum.dtype)
    ex_wts_norm = ex_wts_plus / ex_wts_sum
    # Do not take gradients of the example weights again.
    ex_wts_norm = tf.stop_gradient(ex_wts_norm)

    np.random.seed(0)
    tf.set_random_seed(1234)
    print('Setting tensorflow random seed={:d}'.format(FLAGS.seed))

    with tf.Session() as sess:
        dataset_a = get_noisy_data_inputs(bsize_a, seed=0)

        if bsize_a != bsize_b:
            dataset_b = get_noisy_data_inputs(
                bsize_b, seed=1000)  # Make sure they have different seeds.
        else:
            dataset_b = dataset_a
        data_a = dataset_a.train_noisy
        data_b = dataset_b.train_clean
        data_val = dataset_a.val
        sess.run(tf.global_variables_initializer())
        visual_path = "./visual/test-new-fixed-data-2"
        if not os.path.exists(visual_path):
            os.mkdir(visual_path)
        writer = tf.summary.FileWriter(visual_path, sess.graph)

        max_iter = 80000
        eval_iter = 250
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for iter_num in trange(max_iter):
            if iter_num == 0:
                model_main.adjust_learning_rate(sess, 0.03)
            if iter_num == 40000:
                model_main.adjust_learning_rate(sess, 0.003)
            if iter_num == 60000:
                model_main.adjust_learning_rate(sess, 0.0003)

            train_image_value, train_label_value, meta_image_value, meta_label_value = sess.run(
                [data_a.inputs, data_a.labels, data_b.inputs, data_b.labels])
            ex_wts_value = sess.run(ex_wts_norm, feed_dict={model_a.data: train_image_value,
                                                            model_a.label: train_label_value,
                                                            model_b.data: meta_image_value,
                                                            model_b.label: meta_label_value})
            loss_main_value = model_main.train(sess, data=train_image_value, label=train_label_value,
                                               ex_wts=ex_wts_value)[0]
            tqdm.write(
                "Iter: {}/{} Loss_main: {:2f}".format(iter_num, max_iter, loss_main_value))
            summary = tf.Summary()
            summary.value.add(tag='loss', simple_value=loss_main_value)
            writer.add_summary(summary, iter_num)

            if iter_num % eval_iter == 0:
                acc_num = 0
                val_dataset = NoiseCifar(root="/data/kouzhi/CIFAR_NOISE", method="val", batch_size=batch_size)

                iter_val = iter(val_dataset)
                val_image, val_label = next(iter_val)

                for eval_iter_num in range(50):
                    val_image_value, val_label_value = sess.run(
                        [data_val.inputs, data_val.labels])
                    acc_num += model_main.eval(sess, data=val_image_value, label=val_label_value)[0]
                val_acc = acc_num / 5000
                tqdm.write("Val Acc: {}".format(val_acc))
                summary = tf.Summary()
                summary.value.add(tag='val_acc', simple_value=val_acc)
                writer.add_summary(summary, iter_num)
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    flags = tf.flags
    flags.DEFINE_bool('baseline', False, 'Run non-reweighting baseline')
    flags.DEFINE_bool('eval', False, 'Whether run evaluation only')
    flags.DEFINE_bool('finetune', False, 'Whether to finetune model')
    flags.DEFINE_bool('random_weight', False, 'Use random weights')
    flags.DEFINE_bool('restore', False, 'Whether restore model')
    flags.DEFINE_bool('verbose', True, 'Whether to show logging.INFO')
    flags.DEFINE_float('noise_ratio', 0.4, 'Noise ratio in the noisy training set')
    flags.DEFINE_integer('bsize_a', 100, 'Batch size multiplier for data A')
    flags.DEFINE_integer('bsize_b', 100, 'Batch size multiplier for data B')
    flags.DEFINE_integer('eval_interval', 1000, 'Number of steps between evaluations')
    flags.DEFINE_integer('log_interval', 10, 'Interval for writing loss values to TensorBoard')
    flags.DEFINE_integer('num_clean', 1000, 'Number of clean images in the training set')
    flags.DEFINE_integer('num_val', 5000, 'Number of validation images')
    flags.DEFINE_integer('save_interval', 10000, 'Number of steps between checkpoints')
    flags.DEFINE_integer('seed', 0, 'Random seed for creating the split')
    flags.DEFINE_string('config', None, 'Manually defined config file')
    flags.DEFINE_string('data_root', '/workspace/kouzhi/uber-research/learning-to-reweight-examples/data',
                        'Data folder')
    flags.DEFINE_string('dataset', 'cifar-10', 'Dataset name')
    flags.DEFINE_string('id', None, 'Experiment ID')
    flags.DEFINE_string('results', './results/cifar', 'Saving folder')
    FLAGS = flags.FLAGS
    main()
