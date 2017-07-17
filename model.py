import numpy as np
from scipy.interpolate import LinearNDInterpolator

from bose_hubbard_mft import kinetic_energy


class Model:
    """The model Class."""

    def __init__(
            self,
            name,
            grid_size,
            x_label,
            y_label,
            x_lim,
            y_lim,
            file_name,
            order_parameter
    ):
        self.name = name
        self.grid_size = grid_size
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_label = x_label
        self.y_label = y_label
        self.file_name = file_name
        self.order_parameter = order_parameter
        # load data file
        data = np.load(file_name)
        shape = (grid_size, grid_size)
        # for background plotting
        xx = np.reshape(data[:, 0], shape)
        yy = np.reshape(data[:, 1], shape)
        # color plot
        zz = [
            order_parameter(entry)
            for entry in data[:, 2:]
        ]
        zz = np.reshape(zz, shape)
        self.plot_data = xx, yy, zz
        # interpolation
        # rescale parameter space to (0,1)
        data[:, 0] -= np.min(data[:, 0])
        data[:, 0] /= np.max(data[:, 0])
        data[:, 1] -= np.min(data[:, 1])
        data[:, 1] /= np.max(data[:, 1])
        self._interp = LinearNDInterpolator(data[:, 0:2], data[:, 2:], fill_value=np.nan)

    def gen(self, points):
        """Interpolate data at points."""

        res = [self._interp(p[0], p[1]) for p in points]
        return np.array(res)


BHM = Model(
    name='BHM',
    grid_size=64,
    x_label='$zJ/U$',
    y_label='$\mu/U$',
    x_lim=['0', '0.12'],
    y_lim=['2', '3'],
    file_name='data_64.npy',
    order_parameter=lambda x: kinetic_energy(x, -1.)
)

HM = Model(
    name='HM',
    grid_size=31,
    x_label='$B/J$',
    y_label='$D/J$',
    x_lim=['-0.6', '0.6'],
    y_lim=['-1', '2'],
    file_name='data_31.npy',
    order_parameter=lambda x: x[0] - x[1]
)
