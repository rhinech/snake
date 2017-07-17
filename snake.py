import numpy as np
from numpy import array
from numpy.linalg import norm


class Snake:
    """The active curve class (Kass)."""

    @staticmethod
    def rotate(v, theta):
        """2D rotation of vector v with angle theta."""

        x, y = v
        x1 = np.cos(theta) * x - np.sin(theta) * y
        y1 = np.sin(theta) * x + np.cos(theta) * y
        return array([x1, y1])

    @staticmethod
    def normal_direction(p, p1, p2):
        """Compute normal direction at p in the segment p1->p->p2."""

        e1 = (p1 - p) / norm(p1 - p)
        e2 = Snake.rotate(e1, np.pi / 2.)
        x = np.dot(p2 - p, e1)
        y = np.dot(p2 - p, e2)
        theta = np.arctan2(y, x)
        return Snake.rotate(e1, theta / 2.)

    def __init__(self, nn, xs, ys, alpha, beta, gamma, width, bc):
        """xs: x-coordinates
           ys: y-coordinates"""

        # load the vertices
        self.vertices = []
        for v in zip(xs, ys):
            self.vertices.append(array(v))
        self.vertices = array(self.vertices)
        self.length = self.vertices.shape[0]
        # boundary condition
        assert bc == 'PBC' or bc == 'OBC'
        self._bc = bc
        # width of snake (determining the sensing region)
        self.widths = np.ones((self.length, 1)) * width
        # for neighbour calculations
        id_ = np.arange(self.length)
        self.less1 = np.roll(id_, +1)
        self.less2 = np.roll(id_, +2)
        self.more1 = np.roll(id_, -1)
        self.more2 = np.roll(id_, -2)
        # implicit time-evolution matrix as in Kass
        self._A = alpha * self._alpha_term() + beta * self._beta_term()
        self._gamma = gamma
        self._inv = np.linalg.inv(self._A + self._gamma * np.identity(self.length))
        # the NN for this snake
        self.nn = nn

    def normals(self):
        """Calculate normal direction at each node."""

        res = [
            Snake.normal_direction(
                self.vertices[i], self.vertices[self.less1[i]], self.vertices[self.more1[i]]
            )
            for i in range(self.length)
        ]
        return array(res)

    def gen_samples(self, num_samples):
        """Generate sample for ML near the snake."""

        points = []  # the sample points
        labels = []  # the labels
        whichs = []  # the corresponding node for the sample
        deri_g = []  # the partial derivative to g
        deri_T = []  # the partial derivative to T
        counter = 0
        assert num_samples % self.length == 0
        for i, (v, n) in enumerate(zip(self.vertices, self.normals())):
            for d in np.linspace(-1, 1, num_samples / self.length):
                # geometry
                r = 2 * self.widths[i] * d
                s = v + r * n
                l = array([0.5 * (1. - np.tanh(d)),
                           0.5 * (1. + np.tanh(d))])
                points.append(s)
                labels.append(l)
                whichs.append(i)
                # cal derivatives
                cosh_d = np.cosh(d)
                deri_g.append(1 / (4 * self.widths[i] * cosh_d * cosh_d))
                deri_T.append(d / (2 * self.widths[i] * cosh_d * cosh_d))
                counter += 1
                if counter == num_samples:
                    return array(points), array(labels), array(whichs), array(deri_g), array(deri_T)

    def _alpha_term(self):
        """The arc-length force."""

        res = np.zeros((self.length, self.length))
        range_ = range(1, self.length - 1) if self._bc == 'OBC' else range(self.length)
        for i in range_:
            res[i, i] += 2.
            res[i, self.less1[i]] += -1.
            res[i, self.more1[i]] += -1.
        return res

    def _beta_term(self):
        """The arc-bending force."""

        res = np.zeros((self.length, self.length))
        range_ = range(1, self.length - 1) if self._bc == 'OBC' else range(self.length)
        for i in range_:
            res[i, i] += 6.
            # nn
            res[i, self.less1[i]] += -4.
            res[i, self.more1[i]] += -4.
            # nnn left
            if self._bc == 'OBC' and i == 1:
                # mirror-point method
                res[1, 0] += 2.
                res[1, 1] -= 1.
            else:
                res[i, self.less2[i]] += +1.
            # nnn right
            if self._bc == 'OBC' and i == self.length - 2:
                # mirror-point method
                res[self.length - 2, self.length - 1] += 2.
                res[self.length - 2, self.length - 2] -= 1.
            else:
                res[i, self.more2[i]] += +1.
        return res

    def update(self, forces):
        """Implicit time evolution."""

        if self._bc == 'OBC':
            forces[0] = forces[-1] = 0.
        self.vertices = np.dot(self._inv, self._gamma * self.vertices + forces)
        self.vertices = np.clip(self.vertices, 0., 1.)
        if True:
            forward = self.reshape(array(self.vertices))
            backward = self.reshape(array(self.vertices)[::-1])[::-1]
            self.vertices = (forward + backward) / 2

    def update_widths(self, inc):
        """Explicit update of snake width for each node."""

        self.widths += inc
        # clipping the width to regularize the snake
        self.widths = np.clip(self.widths, 0.03, 0.08)

    def reshape(self, vertices):
        """Make snake smoother."""

        arc_length = 0
        # calculate the total arc-length
        for i in range(1, self.length):
            arc_length += norm(vertices[i] - vertices[i - 1])
        # average length for each segment
        seg_length = arc_length / (self.length - 1)
        if self._bc == 'PBC':
            arc_length += norm(vertices[0] - vertices[-1])
            seg_length = arc_length / self.length
        for i in range(1, self.length - 1 if self._bc == 'OBC' else self.length):
            # normalized tangent direction at node i-1
            tan_direction = vertices[i] - vertices[i - 1]
            tan_direction /= norm(tan_direction)
            # move node i
            vertices[i] = vertices[i - 1] + tan_direction * seg_length
        return vertices
