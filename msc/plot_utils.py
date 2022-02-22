import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_sample(times, sample) -> Figure:
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


def plot_seizure_occurrences_timeline(onsets, patient_name: str, ax=None):
    names = np.arange(len(onsets))
    # Choose some nice levels
    levels = np.tile([-5, 5, -3, 3, -1, 1],
                     int(np.ceil(len(onsets) / 6)))[:len(onsets)]

    # Create figure and plot a stem plot with the date
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title=f"Seizure Events - {patient_name}", xlabel='wall time')

    markerline, stemline, baseline = ax.stem(onsets, levels,
                                             linefmt="C3-", basefmt="k-",
                                             use_line_collection=True)

    plt.setp(markerline, mec="k", mfc="w", zorder=3)

    # Shift the markers to the baseline by replacing the y-data by zeros.
    markerline.set_ydata(np.zeros(len(onsets)))

    # annotate lines
    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    for d, l, r, va in zip(onsets, levels, names, vert):
        ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l) * 3),
                    textcoords="offset points", va=va, ha="right")

    # format xaxis with 4 month intervals
    # ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=4))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y axis and spines
    ax.get_yaxis().set_visible(False)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.margins(y=0.1)


def plot_seizure_intervals_histogram(onsets, patient_name: str, ax=None):
    intervals = np.array([(onsets[i + 1] - onsets[i]).total_seconds() for i in range(len(onsets) - 1)])
    plt.title(f"Inter-seizure Intervals - {patient_name}")
    _, bins = np.histogram(np.log10(intervals + 1))
    plt.hist(intervals, bins=np.logspace(np.log10(min(intervals)), np.log10(max(intervals)), 50),
             label="inter-seizure intervals")
    plt.vlines(np.mean(intervals), 0, 10, colors='r', label='mean')
    plt.vlines(np.var(intervals), 0, 10, colors='orange', label='variance')
    plt.gca().set_xscale("log")
    plt.xlabel("time (s)")
    plt.ylabel("frequency")
    plt.legend()


def add_tsne_to_df(df, data_columns, label_desc=None, random_state=42, **kwargs):
    row_indexer = slice(None) if label_desc is None else df['label_desc'] == label_desc
    X = df.loc[row_indexer, data_columns].to_numpy()
    # calculate t-SNE values of parameters
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=random_state, **kwargs)
    tsne_results = tsne.fit_transform(X)

    # add t-SNE results to df
    df.loc[row_indexer, 'tsne-2d-one'] = tsne_results[:, 0]
    df.loc[row_indexer, 'tsne-2d-two'] = tsne_results[:, 1]

    # return
    return df


def add_pca_to_df(df, data_columns, label_desc=None):
    row_indexer = slice(None) if label_desc is None else df['label_desc'] == label_desc
    X = df.loc[row_indexer, data_columns].to_numpy()
    # calculate PCA values of parameters
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)

    # add t-SNE results to df
    df.loc[row_indexer, 'pca-2d-one'] = pca_results[:, 0]
    df.loc[row_indexer, 'pca-2d-two'] = pca_results[:, 1]

    # return
    return df
