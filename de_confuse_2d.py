import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import BHM, HM
from nn import NN
from snake import Snake

# select model
# HM: spin-1 Heisenberg chain
# BHM: Bose-Hubbard model
MODEL = 'HM'

# hyper-parameters for training
MINI_BATCH_SIZE = 1500
INITIAL_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.9999

# switch for detailed visualization
# with sampled points and ML-forces
PLOT_MORE = True


class Screen:
    """Screen manager."""

    def __init__(self, model, num_nodes, ax):
        """Initialization of visualization."""

        # line1 is the snake
        # line2 is the sample points
        # lines are the external forces on each snake-node
        self._line1 = ax.plot([], [], '-ow')[0]
        if PLOT_MORE:
            self._line2 = ax.plot([], [], 'x')[0]
            self._lines = [
                ax.plot([], [], 'w')[0]
                for _ in range(num_nodes)
            ]
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel(model.x_label, fontsize=16)
        ax.set_ylabel(model.y_label, fontsize=16)
        ax.set_xticklabels(model.x_lim, fontsize=16)
        ax.set_yticklabels(model.y_lim, fontsize=16)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_aspect('equal')
        plt.tight_layout()

    def update(self, snake, sample_points, forces):
        """Update the plots."""

        # assume closed snake
        x_data = np.concatenate([snake.vertices[:, 0], [snake.vertices[0, 0]]])
        y_data = np.concatenate([snake.vertices[:, 1], [snake.vertices[0, 1]]])
        self._line1.set_xdata(x_data)
        self._line1.set_ydata(y_data)
        # detailed visualization
        if PLOT_MORE:
            # plot sample points
            self._line2.set_xdata(sample_points[:, 0])
            self._line2.set_ydata(sample_points[:, 1])
            # plot external forces with magnification factor
            magnification = 50.
            for i in range(snake.length):
                self._lines[i].set_xdata(
                    [
                        snake.vertices[i, 0],
                        snake.vertices[i, 0] + magnification * forces[i, 0]
                    ]
                )
                self._lines[i].set_ydata(
                    [
                        snake.vertices[i, 1],
                        snake.vertices[i, 1] + magnification * forces[i, 1]
                    ]
                )
        plt.gcf().canvas.flush_events()


def epoch(session, snake, screen, learning_rate, gen, draw=False):
    """One epoch of snake crawling."""

    # generate (by snake) sample points for training
    sample_points, labels, which_nodes, dG0_dg, dG0_dT = snake.gen_samples(num_samples=MINI_BATCH_SIZE)
    # clip sample points to prevent false data
    sample_points = np.clip(sample_points, 0., 1.)
    # generate the training data
    data = gen.gen(points=sample_points)
    # step 1: compute the external forces on the current snake
    current_output = snake.nn.eval(session=session, input_=data)
    dC_dG = -np.log(NN.sig(current_output)) + np.log(1. - NN.sig(current_output))
    normals = snake.normals()
    forces1 = np.zeros((snake.length, 2))  # vector
    forces2 = np.zeros((snake.length, 1))  # scalar
    counters = np.zeros(snake.length)
    # for each sample
    boost1 = 0.08
    boost2 = 0.02
    for i, d1, d2, d3 in zip(which_nodes, dC_dG, dG0_dg, dG0_dT):
        # force is minus gradient
        forces1[i] -= boost1 * learning_rate * (d1[0] - d1[1]) * d2 * normals[i]
        forces2[i] -= boost2 * learning_rate * (d1[0] - d1[1]) * d3
        counters[i] += 1.
    # average batch at each node
    for i in range(snake.length):
        forces1[i] /= counters[i]
        forces2[i] /= counters[i]
    # update plot
    if draw:
        screen.update(snake=snake, sample_points=sample_points, forces=forces1)
    # step 2: update NN
    snake.nn.train(session=session, input_=data, answer=labels, learning_rate=learning_rate)
    # step 3: update the current snake location
    snake.update(forces1)
    # step 4: update the current snake widths
    snake.update_widths(forces2)


def __main__():
    """Self-learning phase boundaries in 2D parameter spaces."""

    # TF graph
    with tf.Graph().as_default():
        # TODO: Put your model here, and choose a proper initial snake for it.
        if MODEL == 'BHM':
            # select model
            model = BHM
            # an open snake
            ss = np.linspace(0., 1., 50)
            alpha, beta, gamma = 0.002, 0.4, 0.25
            snake = Snake(
                nn=NN(
                    name='BHM',
                    size_input__layer=80,
                    size_hidden_layer=80,
                    size_output_layer=2,
                    l2_coeff=0.0001,
                    keep_prob=0.8,
                    optimizer='ADAM'
                ),
                xs=0.00 + 0.95 * np.cos(1. * np.pi * ss - np.pi / 2.),
                ys=0.50 + 0.50 * np.sin(1. * np.pi * ss - np.pi / 2.),
                alpha=alpha, beta=beta, gamma=gamma, width=0.06, bc='OBC'
            )
            # misc
            margin = 0.008
            fn = 'track-BHM'
        if MODEL == 'HM':
            # select model
            model = HM
            # a closed snake
            ss = np.arange(0.00, 1.00, 0.02)
            alpha, beta, gamma = 0.002, 0.4, 0.25
            snake = Snake(
                nn=NN(
                    name='HM',
                    size_input__layer=80,
                    size_hidden_layer=80,
                    size_output_layer=2,
                    l2_coeff=0.0001,
                    keep_prob=0.8,
                    optimizer='ADAM'
                ),
                xs=0.60 + 0.20 * np.cos(2. * np.pi * ss - np.pi / 2.),
                ys=0.60 + 0.35 * np.sin(2. * np.pi * ss - np.pi / 2.),
                alpha=alpha, beta=beta, gamma=gamma, width=0.06, bc='PBC'
            )
            # misc
            margin = 0.017
            fn = 'track-HM'
        # initialize graphics
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plot the known phase diagram
        xx, yy, zz = model.plot_data
        plt.imshow(
            np.flipud(zz if MODEL == 'HM' else zz.transpose()),
            extent=[0. - margin, 1. + margin, 0. - margin, 1. + margin]
        )
        # plot the initial snake
        ax.plot(snake.vertices[:, 0], snake.vertices[:, 1], 'wx')
        # set up dynamic plots
        screen = Screen(model=model, num_nodes=len(ss), ax=ax)
        # TF session
        with tf.Session() as session:
            # run TF tensors here
            session.run(tf.global_variables_initializer())
            counter = 0
            dat = []
            while counter < 1000:
                dat.append(snake.vertices)
                epoch(
                    session=session, snake=snake, screen=screen,
                    learning_rate=INITIAL_LEARNING_RATE * np.power(LEARNING_RATE_DECAY, counter),
                    gen=model, draw=(counter % 1 == 0))
                counter += 1
            np.save(fn, dat)


if __name__ == '__main__':
    __main__()
