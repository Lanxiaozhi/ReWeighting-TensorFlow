import os

import tensorflow as tf
from config import train_config
from data.noise_cifar import NoiseCifar
from model.assign_weight_resnet import AssignWeightResNet
from tqdm import trange, tqdm


def main():
    batch_size = 100
    model_config = train_config['resnet_model_config']

    iter_train = iter(NoiseCifar(root="/data/kouzhi/CIFAR_NOISE", method="train", batch_size=batch_size))
    iter_meta = iter(NoiseCifar(root="/data/kouzhi/CIFAR_NOISE", method="meta", batch_size=batch_size))
    train_image, train_label = next(iter_train)
    meta_image, meta_label = next(iter_meta)

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        visual_path = "./visual/test-new"
        if not os.path.exists(visual_path):
            os.mkdir(visual_path)
        writer = tf.summary.FileWriter(visual_path, sess.graph)

        max_iter = 80000
        eval_iter = 250
        for iter_num in trange(max_iter):
            if iter_num == 0:
                model_main.adjust_learning_rate(sess, 0.1)
            if iter_num == 40000:
                model_main.adjust_learning_rate(sess, model_main.learning_rate * 0.1)
            if iter_num == 60000:
                model_main.adjust_learning_rate(sess, model_main.learning_rate * 0.1)

            train_image_value, train_label_value, meta_image_value, meta_label_value = sess.run(
                [train_image, train_label, meta_image, meta_label])
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
                        [val_image, val_label])
                    acc_num += model_main.eval(sess, data=val_image_value, label=val_label_value)[0]
                val_acc = acc_num / 5000
                tqdm.write("Val Acc: {}".format(val_acc))
                summary = tf.Summary()
                summary.value.add(tag='val_acc', simple_value=val_acc)
                writer.add_summary(summary, iter_num)


if __name__ == "__main__":
    main()
