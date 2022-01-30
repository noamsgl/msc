import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_sample(times, sample) -> None:
    # plot
    plt.clf()
    fig = plt.gcf()
    ax: Axes = fig.add_subplot()
    ax.set_xlabel("time (s)")
    for i in range(len(sample)):
        channel = sample[i]
        channel += i
        ax.plot(times, channel)
    # return
    return plt.gcf()
