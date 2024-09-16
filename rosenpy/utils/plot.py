import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap

def dsa():
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(51/256, 1, N)
    vals[:, 1] = np.linspace(106/256, 1, N)
    vals[:, 2] = np.linspace(121/256, 1, N)
    
    Or = ListedColormap(vals)
    
    
    N = 256
    vals1 = np.ones((N, 4))
    vals1[:, 0] = np.linspace(252/256, 1, N)
    vals1[:, 1] = np.linspace(100/256, 1, N)
    vals1[:, 2] = np.linspace(9/256, 1, N)
    Gr = ListedColormap(vals1)
    Grr = Gr.reversed()
    
    top = cm.get_cmap("Greens_r", 24)
    bottom = cm.get_cmap(Grr, 24)
    
    newcolors = np.vstack((top(np.linspace(0, 1, 24)),
                           bottom(np.linspace(0, 1, 24))))
    return ListedColormap(newcolors, name='OrangeGreen')


def __compute_meshgrid(x, y):
    x_min, x_max, y_min, y_max = x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_mesh = np.array([x1.ravel(), x2.ravel()]).T
    return x1, x2, x_mesh

def classification_predictions(x, y, is_binary, nn=None, threshold=0.0, figsize=(12,6), s=15, cmap=dsa()):
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    plt.scatter(x[:, 0], x[:, 1], c=list(np.array(y).ravel()), s=s, cmap=cmap)

    if nn is not None:
        plt.subplot(1, 2, 2, sharex=ax, sharey=ax)

        x1, x2, x_mesh = __compute_meshgrid(x, y)
        y_mesh = nn.predict(x_mesh)
        y_mesh = np.where(y_mesh <= threshold, 0, 1) 
        plt.scatter(x[:, 0], x[:, 1], c=list(np.array(y).ravel()), s=s, cmap=cmap)
        plt.contourf(x1, x2, y_mesh.reshape(x1.shape), cmap=cmap, alpha=0.5)
        
def make_spiral(n_samples, n_class=2, radius=1, laps=1.0, noise=0.0, random_state=None):
    np.random.seed(random_state)
    x = np.zeros((n_samples * n_class, 2))
    y = np.zeros((n_samples * n_class))
    
    pi_2 = 2 * np.math.pi
    points = np.linspace(0, 1, n_samples)
    r = points * radius
    t = points * pi_2 * laps
    for label, delta_t in zip(range(n_class), np.arange(0, pi_2, pi_2/n_class)):
        random_noise = (2 * np.random.rand(n_samples) - 1) * noise
        index = np.arange(label*n_samples, (label+1)*n_samples)
        x[index] = np.c_[r * np.sin(t + delta_t) + random_noise,
                         r * np.cos(t + delta_t) + random_noise]
        y[index] = label
    return x, y.reshape(-1, 1)
