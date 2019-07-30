import numpy as np
import tensorflow as tf


class AssignWeightResNet(object):

    def __init__(self, config, batch_size=100, is_training=True, wts_dict=None, ex_wts=None):
        self._config = config
        self._data_format = config['data_format']
        self._is_training = is_training
        self._bn_update_ops = None
        if wts_dict is not None:
            self._create_new_var = False
            self._wts_dict = wts_dict
        else:
            self._create_new_var = True

        if ex_wts is not None:
            self._ex_wts = ex_wts
        else:
            self._ex_wts = tf.compat.v1.placeholder_with_default(
                tf.compat.v1.ones([batch_size], dtype=tf.float32) / float(batch_size), [batch_size], name='ex_wts')

        data = tf.compat.v1.placeholder(dtype=tf.float32,
                                        shape=[batch_size, config['input_channel'], config['input_height'],
                                               config['input_width']], name='data')
        label = tf.compat.v1.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self._data = data
        self._label = label

        logits = self._build_graph(data)
        loss = self._compute_loss(logits)
        self._output = logits
        self._loss = loss
        self._acc = self._compute_acc()

        if is_training:
            global_step = tf.compat.v1.get_variable(
                dtype=tf.int32,
                shape=[],
                name='global_step',
                trainable=False,
                initializer=tf.constant_initializer(0)
            )
            self._learning_rate = tf.compat.v1.get_variable(
                dtype=tf.float32,
                shape=[],
                name='learning_rate',
                trainable=False,
                initializer=tf.constant_initializer(0.0)
            )

            self._grads_and_vars = self._backward(loss)
            self._train_opt = self._step(global_step=global_step)
            self._global_step = global_step
            self._new_learning_rate = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            self._learning_rate_decay = tf.compat.v1.assign(self._learning_rate, self._new_learning_rate)

    def _build_graph(self, data):
        return self._residual_net(data)

    def _residual_net(self, data):
        config = self.config
        strides = config['strides']
        dropout = config['dropout']
        if config.get('dilations') is None:
            dilations = [1] * len(config['strides'])
        else:
            dilations = config['dilations']
        assert len(config['strides']) == len(dilations), 'Need to pass in lists of same size.'
        filters = [ff for ff in config['num_filters']]  # Copy filter config.
        init_filter_size = config['init_filter_size']
        if self.data_format == 'NCHW':
            # inp = tf.transpose(inp, [0, 3, 1, 2])
            pass
        with tf.variable_scope('init'):
            h = self._conv('init_conv', data, init_filter_size, self.config['num_channels'], filters[0],
                           self._stride_arr(config['init_stride']), 1)
            h = self._batch_norm('init_bn', h)
            h = self._relu('init_relu', h)

            # Max-pooling is used in ImageNet experiments to further reduce
            # dimensionality.
            if config['init_max_pool']:
                h = tf.nn.max_pool(
                    h,
                    self._stride_arr(3),
                    self._stride_arr(2),
                    'SAME',
                    data_format=self.data_format)

        if config['use_bottleneck']:
            res_func = self._bottleneck_residual
            # For CIFAR-10 it's [16, 16, 32, 64] => [16, 64, 128, 256]
            for ii in range(1, len(filters)):
                filters[ii] *= 4
        else:
            res_func = self._residual

        # New version, single for-loop. Easier for checkpoint.
        nlayers = sum(config['num_residual_units'])
        ss = 0
        ii = 0
        for ll in range(nlayers):
            # Residual unit configuration.
            if ii == 0:
                if ss == 0:
                    no_activation = True
                else:
                    no_activation = False
                in_filter = filters[ss]
                stride = self._stride_arr(strides[ss])
            else:
                in_filter = filters[ss + 1]
                stride = self._stride_arr(1)

            # Compute out filters.
            out_filter = filters[ss + 1]

            # Compute dilation rates.
            if dilations[ss] > 1:
                if config['use_bottleneck']:
                    dilation = [dilations[ss] // strides[ss], dilations[ss], dilations[ss]]
                else:
                    dilation = [dilations[ss] // strides[ss], dilations[ss]]
            else:
                if config['use_bottleneck']:
                    dilation = [1, 1, 1]
                else:
                    dilation = [1, 1]

            # Build residual unit.
            with tf.variable_scope('unit_{}_{}'.format(ss + 1, ii)):
                h = res_func(
                    h,
                    in_filter,
                    out_filter,
                    stride,
                    dilation,
                    dropout=dropout,
                    no_activation=no_activation)

            if (ii + 1) % config['num_residual_units'][ss] == 0:
                ss += 1
                ii = 0
            else:
                ii += 1

        # Make a single tensor.
        if type(h) == tuple:
            h = tf.concat(h, axis=3)

        # Classification layer.
        if config['build_classifier']:
            with tf.variable_scope('unit_last'):
                h = self._batch_norm('final_bn', h)
                h = self._relu('final_relu', h)

            h = self._global_avg_pool(h)
            with tf.variable_scope('logit'):
                h = self._fully_connected(h, config['num_classes'])

        return h

    def _weight_variable(self,
                         shape,
                         init_method=None,
                         dtype=tf.float32,
                         init_param=None,
                         weight_decay=None,
                         name=None,
                         trainable=True,
                         seed=0):
        """
        A wrapper to declare variables on CPU.

        See nnlib.py:weight_variable_cpu for documentation.
        """
        var = self._weight_variable_cpu(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            weight_decay=weight_decay,
            name=name,
            trainable=trainable,
            seed=seed)
        if self._create_new_var:
            if self.wts_dict is None:
                self.wts_dict = {}
            self.wts_dict[var.name] = var
            return var
        else:
            return self.wts_dict[var.name]

    @staticmethod
    def _weight_variable_cpu(shape,
                             init_method=None,
                             dtype=tf.float32,
                             init_param=None,
                             weight_decay=None,
                             name=None,
                             trainable=True,
                             seed=0):
        with tf.device('/cpu:0'):
            if init_method is None:
                initializer = tf.zeros_initializer(dtype=dtype)
            elif init_method == 'truncated_normal':
                if 'mean' not in init_param:
                    mean = 0.0
                else:
                    mean = init_param['mean']
                if 'stddev' not in init_param:
                    stddev = 0.1
                else:
                    stddev = init_param['stddev']
                # log.info('Normal initialization std {:.3e}'.format(stddev))
                initializer = tf.truncated_normal_initializer(
                    mean=mean, stddev=stddev, seed=seed, dtype=dtype)
            elif init_method == 'uniform_scaling':
                if 'factor' not in init_param:
                    factor = 1.0
                else:
                    factor = init_param['factor']
                # log.info('Uniform initialization scale {:.3e}'.format(factor))
                initializer = tf.uniform_unit_scaling_initializer(factor=factor, seed=seed, dtype=dtype)
            elif init_method == 'constant':
                if 'val' not in init_param:
                    value = 0.0
                else:
                    value = init_param['val']
                initializer = tf.constant_initializer(value=value, dtype=dtype)
            elif init_method == 'xavier':
                initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=seed, dtype=dtype)
            else:
                raise ValueError('Non supported initialization method!')

            if weight_decay is not None:
                if weight_decay > 0.0:

                    def _reg(x):
                        return tf.multiply(tf.nn.l2_loss(x), weight_decay)

                    reg = _reg
                else:
                    reg = None
            else:
                reg = None
            var = tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer=reg, dtype=dtype,
                                  trainable=trainable)
            return var

    def _stride_arr(self, stride):
        """
        Map a stride scalar to the stride array for tf.nn.conv2d.

        :param stride: [int] Size of the stride.

        :return:       [list] [1, stride, stride, 1]
        """
        if self.data_format == 'NCHW':
            return [1, 1, stride, stride]
        else:
            return [1, stride, stride, 1]

    def _batch_norm(self, name, x):
        """
        Applies batch normalization.

        :param name:    [string]    Name of the variable scope.
        :param x:       [Tensor]    Tensor to apply BN on.
        :param add_ops: [bool]      Whether to add BN updates to the ops list, default True.

        :return:        [Tensor]    Normalized activation.
        """
        bn = tf.contrib.layers.batch_norm(
            x, fused=True, data_format=self.data_format, is_training=self.is_training)
        return bn

    def _possible_downsample(self, x, in_filter, out_filter, stride):
        """
        Downsamples the feature map using average pooling, if the filter size
        does not match.

        :param x:             [Tensor]     Input to the downsample.
        :param in_filter:     [int]        Input number of channels.
        :param out_filter:    [int]        Output number of channels.
        :param stride:        [list]       4-D strides array.

        :return:              [Tensor]     Possibly downsampled activation.
        """
        if stride[2] > 1:
            with tf.variable_scope('downsample'):
                x = tf.nn.avg_pool(x, stride, stride, 'SAME', data_format=self.data_format)

        if in_filter < out_filter:
            with tf.variable_scope('pad'):
                pad_ = [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
                if self.data_format == 'NCHW':
                    x = tf.pad(x, [[0, 0], pad_, [0, 0], [0, 0]])
                else:
                    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], pad_])
        return x

    def _residual_inner(self,
                        x,
                        in_filter,
                        out_filter,
                        stride,
                        dilation_rate,
                        dropout=0.0,
                        no_activation=False):
        """
        Inner transformation applied on residual units.

        :param x:              [Tensor]     Input to the residual function.
        :param in_filter:      [int]        Input number of channels.
        :param out_filter:     [int]        Output number of channels.
        :param stride:         [list]       4-D strides array.
        :param dilation_rate:  [list]       List of 2 integers, dilation rate for each conv.
        :param dropout:        [float]      Whether to dropout in the middle.
        :param no_activation:  [bool]       Whether to have BN+ReLU in the first.

        :return:               [Tensor]     Output of the residual function.
        """
        with tf.variable_scope('sub1'):
            if not no_activation:
                x = self._batch_norm('bn1', x)
                x = self._relu('relu1', x)
                if dropout > 0.0 and self.is_training:
                    print('Using dropout with {:d}%'.format(int(dropout * 100)))
                    x = tf.nn.dropout(x, keep_prob=(1.0 - dropout))
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride, dilation_rate[0])
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu('relu2', x)
            x = self._conv('conv2', x, 3, out_filter, out_filter,
                           self._stride_arr(1), dilation_rate[1])
        return x

    def _residual(self,
                  x,
                  in_filter,
                  out_filter,
                  stride,
                  dilation_rate,
                  dropout=0.0,
                  no_activation=False):
        """
        A residual unit with 2 sub layers.

        :param x:              [tf.Tensor]  Input to the residual unit.
        :param in_filter:      [int]        Input number of channels.
        :param out_filter:     [int]        Output number of channels.
        :param stride:         [list]       4-D strides array.
        :param dilation_rate   [list]       List of 2 integers, dilation rate for each conv.
        :param no_activation:  [bool]       Whether to have BN+ReLU in the first.

        :return:               [tf.Tensor]  Output of the residual unit.
        """
        orig_x = x
        x = self._residual_inner(
            x,
            in_filter,
            out_filter,
            stride,
            dilation_rate,
            dropout=dropout,
            no_activation=no_activation)
        x += self._possible_downsample(orig_x, in_filter, out_filter, stride)
        return x

    def _bottleneck_residual_inner(self,
                                   x,
                                   in_filter,
                                   out_filter,
                                   stride,
                                   dilation_rate,
                                   no_activation=False):
        """
        Inner transformation applied on residual units (bottleneck).

        :param x:              [Tensor]     Input to the residual function.
        :param in_filter:      [int]        Input number of channels.
        :param out_filter:     [int]        Output number of channels.
        :param stride:         [list]       4-D strides array.
        :param dilation_rate   [list]       List of 3 integers, dilation rate for each conv.
        :param no_activation:  [bool]       Whether to have BN+ReLU in the first.

        :return:               [Tensor]     Output of the residual function.
        """
        with tf.variable_scope('sub1'):
            if not no_activation:
                x = self._batch_norm('bn1', x)
                x = self._relu('relu1', x)
            x = self._conv('conv1', x, 1, in_filter, out_filter // 4, stride, dilation_rate[0])
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu('relu2', x)
            x = self._conv('conv2', x, 3, out_filter // 4, out_filter // 4,
                           self._stride_arr(1), dilation_rate[1])
        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu('relu3', x)
            x = self._conv('conv3', x, 1, out_filter // 4, out_filter,
                           self._stride_arr(1), dilation_rate[2])
        return x

    def _possible_bottleneck_downsample(self, x, in_filter, out_filter, stride):
        """Downsample projection layer, if the filter size does not match.

        :param x:             [Tensor]     Input to the downsample.
        :param in_filter:     [int]        Input number of channels.
        :param out_filter:    [int]        Output number of channels.
        :param stride:        [list]       4-D strides array.
        :param dilation_rate: [int]        Dilation rate.

        :return:              [Tensor]     Possibly downsampled activation.
        """
        if stride[1] > 1 or in_filter != out_filter:
            x = self._conv('project', x, 1, in_filter, out_filter, stride, 1)
        return x

    def _bottleneck_residual(self,
                             x,
                             in_filter,
                             out_filter,
                             stride,
                             dilation_rate,
                             no_activation=False):
        """
        A bottleneck resisual unit with 3 sub layers.

        :param x:              [Tensor]    Input to the residual unit.
        :param in_filter:      [int]       Input number of channels.
        :param out_filter:     [int]       Output number of channels.
        :param stride:         [list]      4-D strides array.
        :param dilation_rate   [list]      List of 3 integers, dilation rate for each conv.
        :param no_activation:  [bool]      Whether to have BN+ReLU in the first.

        :return:               [Tensor]    Output of the residual unit.
        """
        orig_x = x
        x = self._bottleneck_residual_inner(
            x, in_filter, out_filter, stride, dilation_rate, no_activation=no_activation)
        x += self._possible_bottleneck_downsample(orig_x, in_filter, out_filter, stride)
        return x

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, dilation_rate):
        """Convolution.

        :param name           [string]     Name of the op.
        :param x:             [Tensor]     Input to the downsample.
        :param filter_size    [list]       4-D kernel shape.
        :param in_filter:     [int]        Input number of channels.
        :param out_filter:    [int]        Output number of channels.
        :param stride:        [list]       4-D strides array.
        :param dilation_rate: [int]        Convolution dilation rate.

        :return:              [Tensor]     Convolution output.
        """
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            init_method = 'truncated_normal'
            init_param = {'mean': 0, 'stddev': np.sqrt(2.0 / n)}
            kernel = self._weight_variable(
                [filter_size, filter_size, in_filters, out_filters],
                init_method=init_method,
                init_param=init_param,
                weight_decay=self.config['weight_decay'],
                dtype=tf.float32,
                name='w')
            if dilation_rate == 1:
                return tf.nn.conv2d(
                    x, kernel, strides, padding='SAME', data_format=self.data_format)
            elif dilation_rate > 1:
                assert self.data_format == 'NHWC', 'Dilated convolution needs to be in NHWC format.'
                assert all([strides[ss] == 1 for ss in range(len(strides))]), 'Strides need to be 1'
                return tf.nn.atrous_conv2d(x, kernel, dilation_rate, padding='SAME')

    def _relu(self, name, x):
        """
        Applies ReLU function.

        :param name: [string]     Name of the op.
        :param x:    [Tensor]     Input to the function.

        :return:     [Tensor]     Output of the function.
        """
        return tf.nn.relu(x, name=name)

    def _fully_connected(self, x, out_dim):
        """
        A FullyConnected layer for final output.

        :param x:         [Tensor]     Input to the fully connected layer.
        :param out_dim:   [int]        Number of output dimension.

        :return:          [Tensor]     Output of the fully connected layer.
        """
        x_shape = x.get_shape()
        d = x_shape[1]
        w = self._weight_variable(
            [d, out_dim],
            init_method='uniform_scaling',
            init_param={'factor': 1.0},
            weight_decay=self.config['weight_decay'],
            dtype=tf.float32,
            name='w')
        b = self._weight_variable(
            [out_dim], init_method='constant', init_param={'val': 0.0}, name='b', dtype=tf.float32)
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        """
        Applies global average pooling.

        :param x:  [Tensor]   Input to average pooling.

        :return:   [Tensor]   Pooled activation.
        """
        if self.data_format == 'NCHW':
            return tf.reduce_mean(x, [2, 3])
        else:
            return tf.reduce_mean(x, [1, 2])

    def _compute_loss(self, logits):
        with tf.compat.v1.variable_scope('loss'):
            item_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.label,
                                                               reduction=tf.losses.Reduction.NONE)
            wts_loss = tf.compat.v1.reduce_sum(item_loss * self.ex_wts, name='wts_loss') + self._decay()
            self.cross_ent_loss = tf.compat.v1.reduce_mean(item_loss)
        return wts_loss

    def _compute_acc(self):
        output_idx = tf.cast(tf.argmax(self.output, axis=1), self.label.dtype)
        return tf.reduce_sum(tf.to_float(tf.equal(output_idx, self.label)))

    def adjust_learning_rate(self, sess, new_learning_rate):
        sess.run(self._learning_rate_decay, feed_dict={self._new_learning_rate: new_learning_rate})

    def _backward(self, loss, var_list=None):
        if var_list is None:
            var_list = tf.compat.v1.trainable_variables()
        grads = tf.gradients(loss, var_list)
        return zip(grads, var_list)

    def _step(self, global_step):
        opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        return opt.apply_gradients(self._grads_and_vars, global_step, name='train_step')

    def _decay(self):
        weight_decay_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # print('Weight decay variables')
        # [print(x) for x in weight_decay_losses]
        print('Total length: {}'.format(len(weight_decay_losses)))
        if len(weight_decay_losses) > 0:
            return tf.add_n(weight_decay_losses)
        else:
            return 0.0

    def train(self, sess, data, label, ex_wts=None):
        return sess.run(
            [self.loss, self._train_opt] + self.bn_update_ops,
            feed_dict={self.data: data, self.label: label, self.ex_wts: ex_wts}
        )

    def eval(self, sess, data, label):
        return sess.run(
            [self.acc, self.loss], feed_dict={self.data: data, self.label: label}
        )

    @property
    def config(self):
        return self._config

    @property
    def is_training(self):
        return self._is_training

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def output(self):
        return self.output

    @property
    def loss(self):
        return self._loss

    @property
    def wts_dict(self):
        return self._wts_dict

    @wts_dict.setter
    def wts_dict(self, wts_dict):
        self._wts_dict = wts_dict

    @property
    def ex_wts(self):
        return self._ex_wts

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def acc(self):
        return self._acc

    @property
    def bn_update_ops(self):
        return self._bn_update_ops

    @property
    def data_format(self):
        return self._data_format
