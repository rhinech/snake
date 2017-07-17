import numpy as np
import tensorflow as tf


class NN:
    """TF NN class with one hidden layer."""

    def __init__(self,
                 name,
                 size_input__layer,
                 size_hidden_layer,
                 size_output_layer,
                 l2_coeff,
                 keep_prob,
                 optimizer='SGD'):
        """Make new tensors and connect them."""

        self.size_input__layer = size_input__layer
        self.size_hidden_layer = size_hidden_layer
        self.size_output_layer = size_output_layer
        self.keep_prob = keep_prob
        self.input__placeholder = tf.placeholder(tf.float32, shape=(None, size_input__layer))
        self.answer_placeholder = tf.placeholder(tf.float32, shape=(None, size_output_layer))
        self.learning_rate = tf.placeholder(tf.float32)
        self.inference_proc, l2_proc = self.inference(self.input__placeholder, name)
        self.loss_proc = NN.loss(self.inference_proc, l2_proc, l2_coeff, self.answer_placeholder)
        self.training_proc = NN.training(self.loss_proc, self.learning_rate, optimizer)

    def eval(self, session, input_):
        """Run inference procedure."""

        return session.run(
            self.inference_proc,
            feed_dict={
                self.input__placeholder: input_
            })

    def train(self, session, input_, answer, learning_rate):
        """Run training procedure"""

        session.run(
            self.training_proc,
            feed_dict={
                self.input__placeholder: input_,
                self.answer_placeholder: answer,
                self.learning_rate: learning_rate
            })

    def inference(self, input_, name):
        """Make the NN w/o final non-linearity."""

        # if exists then reuse
        with tf.variable_scope(name, reuse=False):
            with tf.name_scope('hidden'):
                weights = tf.get_variable(
                    name='w1',
                    shape=(self.size_input__layer, self.size_hidden_layer),
                    initializer=tf.random_normal_initializer(stddev=1. / np.sqrt(self.size_input__layer))
                )
                biases = tf.get_variable(
                    name='b1',
                    shape=self.size_hidden_layer,
                    initializer=tf.constant_initializer()
                )
                hidden = tf.nn.sigmoid(tf.matmul(input_, weights) + biases)
                l2a = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
            with tf.name_scope('output'):
                weights = tf.get_variable(
                    name='w2',
                    shape=[self.size_hidden_layer, self.size_output_layer],
                    initializer=tf.random_normal_initializer(stddev=1. / np.sqrt(self.size_hidden_layer))
                )
                biases = tf.get_variable(
                    name='b2',
                    shape=self.size_output_layer,
                    initializer=tf.constant_initializer()
                )
                # with dropout at hidden layer
                output = tf.matmul(tf.nn.dropout(hidden, keep_prob=self.keep_prob), weights) + biases
                l2b = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
            return output, l2a + l2b

    @staticmethod
    def loss(output, l2_proc, l2_coeff, answer):
        """Make loss operator."""

        cross_entropy_without_l2 = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=answer,
                logits=output
            ),
            reduction_indices=[1]
        )
        loss_proc = tf.reduce_mean(cross_entropy_without_l2) + l2_coeff * l2_proc
        return loss_proc

    @staticmethod
    def training(loss_proc, learning_rate, optimizer_):
        """Make training operator."""

        # TODO: tweak optimizer here, ADAM does not work for Ising, why?
        if optimizer_ == 'ADAM':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_proc = optimizer.minimize(loss_proc)
        return train_proc

    @staticmethod
    def sig(x):
        """Sigmoid function."""

        return 1. / (np.exp(-x) + 1.)
