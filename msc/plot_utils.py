import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def square_size(length):
        """set figure dimensions to square"""
        inches_per_pt = 1 / 72.27
        fig_width_in = length * inches_per_pt
        fig_dim = (fig_width_in, fig_width_in)
        return fig_dim

def set_size(width, fraction=1., height_scale=1., transposed=False):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    height_scale: float, optional
            Fraction of the golden_ratio you wish the aspect of the figure to have

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * height_scale

    fig_dim = (fig_width_in, fig_height_in)

    if transposed:
        fig_dim = fig_dim[::-1]
        
    return fig_dim
    

def plot_sample(times, sample, yfactor=10, ax=None) -> Figure:
    # plot
    sample = sample.T
    if ax is None:
        plt.clf()
        fig = plt.gcf()
        ax: Axes = fig.add_subplot()
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("EEG")
    ax.set_yticks([])
    for i in range(len(sample)):
        channel = sample[i]
        channel += yfactor * i
        ax.plot(times, channel)
    # return
    return plt.gcf()
