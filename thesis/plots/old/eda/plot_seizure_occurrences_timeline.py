import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from msc.dataset import get_seizures_index_df


def plot_seizure_occurrences_timeline(selected_patient, seizures_index_df, ax=None):
    patient_seizures_df = seizures_index_df.loc[seizures_index_df['patient'] == selected_patient]

    dates = patient_seizures_df.onset
    names = patient_seizures_df.seizure_num

    # Choose some nice levels
    levels = np.tile([-5, 5, -3, 3, -1, 1],
                     int(np.ceil(len(dates) / 6)))[:len(dates)]

    # Create figure and plot a stem plot with the date
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title=f"Patient {selected_patient} Seizure Events", xlabel='wall time')

    markerline, stemline, baseline = ax.stem(dates, levels,
                                             linefmt="C3-", basefmt="k-",
                                             use_line_collection=True)

    plt.setp(markerline, mec="k", mfc="w", zorder=3)

    # Shift the markers to the baseline by replacing the y-data by zeros.
    markerline.set_ydata(np.zeros(len(dates)))

    # annotate lines
    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    for d, l, r, va in zip(dates, levels, names, vert):
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


if __name__ == '__main__':
    seizures_index_df = get_seizures_index_df()
    selected_patient = 'pat_3700'
    plot_seizure_occurrences_timeline(selected_patient, seizures_index_df)
    plt.show()
