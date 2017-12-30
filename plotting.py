"""Functions to create plots."""

from math import ceil
import numpy as np
from scipy.stats import gaussian_kde, pearsonr, linregress
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, LogLocator

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams.update({'font.size': 10})


def hist2d(fig, ax, x, y, i, show_slope=False, colorbar=False):
    """
    Plot a 2D histogram of data with equalized axes.

    Parameters
    ----------
    fig : plt.Figure
        Figure to place plot
    ax : plt.Axes
        Axis to place plot
    x, y : array, shape (n,)
        Data to be histogrammed
    i : int
        Layer number
    show_slope : bool
        Show slope in plot title
    colorbar : bool
        Show colorbar

    Returns
    -------
    lr :
        Linear regression results
    """
    cax = ax.hist2d(x, y, bins=50, norm=LogNorm(), normed=True,
                    cmap='Greens', zorder=2)[-1]
    r, p = pearsonr(x, y)
    lr = linregress(x, y)
    slope = lr.slope
    intercept = lr.intercept

    mm = np.max([-x.min(), x.max()])
    x0, x1 = - mm, mm

    ax.set_xlim([x0, x1])
    ax.set_ylim([x0 * slope, x1 * slope])
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    xx = np.arange(x0, x1 + 0.1 * (x1 - x0), (x1 - x0) / 100)
    yy = xx * slope + intercept
    ax.plot(xx, yy, 'k--', zorder=3)

    if show_slope:
        ax.set_title(
            'r = {:.2f}, slope = {:.2f}, Layer {}'.format(r, slope, i))
    else:
        ax.set_title('Layer {} : r = {:.2f}'.format(i, r))
    if colorbar:
        fig.colorbar(cax, ax=ax)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    return lr


def kde_scipy(x, x_grid, bandwidth=0.02, **kwargs):
    """Kernel Density Estimation with Scipy."""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def _angle(a, b):
    """Return the angle between two vectors."""
    return np.arccos(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def get_angles(x, y):
    """
    Finds the angles between each pair in two lists of arrays

    Parameters
    ----------
    x : array or list
        array or list of vectors
    y : array or list
        array or list of vectors

    Returns
    -------
    angles : list
        angles (in radians) between every vector pair in the two input lists/arrays
    """
    return np.array([_angle(a, b) for a, b in zip(x, y)])


def corr_plot(x_, y_, x_label=None, y_label=None, save_path=None):
    """
    Plots a 2d histogram of the elements in two arrays,

    Parameters
    ----------
    b_ : list of arrays
        List of data for horizontal axis
    c_ : list of arrays
        List of data for vertical axis
    x_label, y_label : str
        Labels for x, y axes
    data_set : str
        data set being used; used for the save file
    """

    num = len(x_)
    cols = 3
    rows = int(ceil(num/(1.0 * cols)))
    size = (cols * 7./3., rows * 4/2.)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=size)
    for i, (x, y) in enumerate(zip(x_, y_)):
        ax = axes.flat[i]
        colorbar = ((i % cols) == cols - 1) & (i / cols == 0)

        hist2d(fig, ax, x.ravel(), y.ravel(), i+1, colorbar=colorbar)
        fontsize = 12
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)

        ymin, ymax = ax.get_ylim()
        ax.axhspan(ymin, 0, xmin=0.5, alpha=0.5, color='gray', zorder=1)
        ax.axhspan(0, ymax, xmax=0.5, alpha=0.5, color='gray', zorder=1)

    # Remove excess labels
    for i in range(rows):
        for j in range(cols):
            ax = axes[i][j]
            if i != rows-1:
                ax.set_xlabel('')
            if j != 0:
                ax.set_ylabel('')

    # Delete extra plots
    if num < (rows*cols):
        for i in range((rows*cols)-num):
            fig.delaxes(axes.flat[rows*cols-1-i])
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    return fig, axes


def weight_dist(w_, save_path=None, yscale='log'):
    """
    Plots log distributions of continuous weights in a binary neural network;
    saves output

    Parameters
    ----------
    w_ : list of arrays
            arrays are continous weights
    data_set : str
        data set being used; used for the save file
    """
    x_grid = np.arange(-1, 1, 0.02)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 2),
                             sharex=True, sharey=True)
    for i, w in enumerate(w_):
        pdf = kde_scipy(w.ravel(), x_grid, bandwidth=.1)
        ax = axes
        ax.plot(x_grid, pdf, lw=1, alpha=1.0, label='Layer {}'.format(i+1))
    ax.set_ylim([0.00001, 3])
    ax.xaxis.set_major_locator(MaxNLocator(3))
    if yscale == 'log':
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    elif yscale == 'linear':
        ax.yaxis.set_major_locator(MaxNLocator(3))
    else:
        raise ValueError('Invalid yscale')
    ax.set_yscale(yscale)
    plt.legend(loc=8, labelspacing=0.1, prop={'size': 8})
    plt.title('Weight Histogram')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fig, ax


def angle_plot(x, y, data_set, save_path=None, zoom=True,
               exp=np.arccos(np.sqrt(2 / np.pi))):
    """
    Plots the distribution of angles between every pair in two lists
    of vectors

    Parameters
    ----------
    x : array or list
        array or list of vectors
    y : array or list
        array or list of vectors
    data_set : str
        for file naming purposes
    save_path : str
        Path to save plot
    exp : float
        Plot a vertical line at this value
    zoom : bool
        If true, zoom the x-axis to show deviations from theory
    """

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    if zoom:
        x_grid = np.arange(0, np.pi/2., 0.005)
    else:
        x_grid = np.arange(0, np.pi, 0.005)
    for i, (c, b) in enumerate(zip(x, y)):
        angles = get_angles(c, b)
        pdf = kde_scipy(angles, x_grid, bandwidth=0.01)
        ax.plot(x_grid, pdf, label='L {}, d={}'.format(i+1, len(c[0])),
                alpha=0.5)
    ax.axvline(x=exp, color='black', ls='--', label='d = $\infty$')
    ax.tick_params(labelsize=10)
    if zoom:
        ax.set_xlim([np.pi/16, np.pi/4])
        ax.set_xticks([np.pi/16, np.pi/4])
        ax.set_xticklabels([r'$\pi/16$', r'$\pi/4$'])
    else:
        ax.set_xlim([0., np.pi])
        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([])
    ax.legend(loc='best', labelspacing=0.1, prop={'size': 7})
    ax.set_title(r'$\angle (w^b, w^c)$ for {}'.format(data_set))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fig, ax


def stddev_angles(x, y, data_set, save_path=None):
    """
    Plots the distribution of angles between every pair in two lists
    of vectors

    Parameters
    ----------
    x : array or list
        array or list of vectors
    y : array or list
        array or list of vectors
    data_set : str
        for file naming purposes
    save : bool
        determine whether or not the plot should be saved or not
    """
    std_angles = []
    n = []
    for i, (c, b) in enumerate(zip(x, y)):
        angles = get_angles(c, b)
        std_angles.append(np.std(angles))
        n.append(len(c[0]))

    std_angles = np.log(std_angles)
    n = np.log(n)

    output = linregress(n, std_angles)
    #  slope = output[0]
    #  intercept = output[1]

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.plot(n, std_angles, 'bo')

    x_t = np.arange(n[0], n[-1] + 0.1, .1)
    y_t = -0.5*(x_t)

    bias = (std_angles + 0.5 * n).mean()
    y_t += bias

    ax.plot(x_t, y_t, 'k--', lw=2, label=r'$d^{-0.5}$')

#     ax.set_ylabel(r'$\log \, \sigma(d)$')
#     ax.set_xlabel(r'$\log\, d$')

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    ax.legend()
#     plt.title('$d$ vs. $\sigma(d)$ for {}\n slope = {:.2f}'.format(data_set, slope))
    ax.set_title(r'$\log\, d$ vs $\log\, \sigma(d)$')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fig, ax


def show_fields(d, cmap=plt.cm.gray, m=None, pos_only=False,
                colorbar=True):
    """
    Plot a collection of images.

    Parameters
    ----------
    d : array, shape (n, n_pix)
        A collection of n images unrolled into n_pix length vectors
    cmap : plt.cm
        Color map for plot
    m : int
        Plot a m by m grid of receptive fields
    """
    n, n_pix = d.shape
    if m is None:
        m = int(np.sqrt(n - 0.01)) + 1

    l = int(np.sqrt(n_pix))  # Linear dimension of the image

    mm = np.max(np.abs(d))

    out = np.zeros(((l + 1) * m - 1, (l + 1) * m - 1)) + mm

    for u in range(n):
        i = u / m
        j = u % m
        out[(i * (l + 1)):(i * (l + 1) + l),
            (j * (l + 1)):(j * (l + 1) + l)] = np.reshape(d[u], (l, l))

    if pos_only:
        m0 = 0
    else:
        m0 = -mm
    m1 = mm
    plt.imshow(out, cmap=cmap, interpolation='nearest', vmin=m0, vmax=m1)
    if colorbar:
        plt.colorbar()
    plt.axis('off')
