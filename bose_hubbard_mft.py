from functools import partial

import numpy as np
from scipy.optimize import minimize

# constants
DIM = 1
INTERACTION = 1.

# data size
CUTOFF = 80
GRID_SIZE = 64


def kinetic_energy(fs, hopping):
    """Mean-field kinetic energy."""
    return -DIM * hopping * np.square(
        np.sum(np.sqrt(n + 1.) * fs[n] * fs[n + 1] for n in range(len(fs) - 1))
    )


def num_particles(fs):
    """Mean-field occupation."""
    return np.sum(n * fs[n] * fs[n] for n in range(len(fs)))


def on_site_energy(fs, mu):
    """Mean-field on-site energy."""
    return -mu * num_particles(fs)


def interaction_energy(fs):
    """Mean-field Hubbard energy."""
    return INTERACTION / 2. * np.sum(n * (n - 1.) * fs[n] * fs[n] for n in range(len(fs)))


def energy_per_site(fs, hopping, mu):
    """Mean-field total energy per site."""
    return (kinetic_energy(fs, hopping) + on_site_energy(fs, mu) + interaction_energy(fs)) / DIM


def constraint_normalization(fs):
    """Normalization condition of wave-function."""
    return np.square(fs).sum() - 1.


def init_fs(cutoff, kappa):
    """The kappa trial wave-function as initial state."""
    res = np.array([
        np.exp(-kappa * n * n / 2.) / np.sqrt(float(np.math.factorial(n)))
        for n in range(cutoff)
    ])
    res /= np.linalg.norm(res)
    return res


def optimize(p1, p2):
    """Find mean-field state for J/U=p1 and mu/U=p2."""
    init = init_fs(cutoff=CUTOFF, kappa=1.)
    # the bound is crucial for convergence
    res = minimize(
        partial(energy_per_site, hopping=p1, mu=p2),
        init,
        bounds=[[0., 1.]] * CUTOFF,
        constraints=[
            {'type': 'eq', 'fun': constraint_normalization},
        ])
    return res.x


def generate_data():
    """Generate grid of data for interpolation."""
    res = []
    for hopping in np.linspace(0.0, 0.12, GRID_SIZE):
        for mu in np.linspace(2.0, 3.0, GRID_SIZE):
            print(hopping, mu)
            res.append(np.concatenate([[hopping, mu], optimize(hopping, mu)]))
    res = np.array(res)
    np.save(r'data_%d' % GRID_SIZE, np.array(res))


def load_data():
    """Draw the Mott lobes."""

    res = np.load(r'data_%d.npy' % GRID_SIZE)
    x = res[:, 0]
    y = res[:, 1]
    z = []
    for i, entry in enumerate(res):
        z.append(kinetic_energy(entry[2:], -1.))
    plt.pcolor(
        np.reshape(x, (GRID_SIZE, GRID_SIZE)),
        np.reshape(y, (GRID_SIZE, GRID_SIZE)),
        np.reshape(z, (GRID_SIZE, GRID_SIZE))
    )
    plt.xlabel('$dt/U$')
    plt.ylabel('$\mu/U$')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for i, J in enumerate(np.linspace(0, 0.12, 9)):
        plt.subplot(3, 3, i + 1)
        fs = optimize(J, 2.95)
        plt.plot(fs, '-o')
        plt.xlim([0, 10])
    plt.tight_layout()
    plt.show()
