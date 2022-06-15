import matplotlib.pyplot as plt

from msc.dataset import get_seizures_index_df


def plot_seizure_count_histogram(seizures_index_df, ax=None):
    if ax is None:
        ax = plt.gca()
    seizures_index_df.groupby('patient').size().plot(kind='hist', grid=True,
                                                     title='Distribution of Seizure Counts per Patient', ax=ax)
    ax.set_xlabel('Count of Seizures per Patient')


if __name__ == '__main__':
    seizures_index_df = get_seizures_index_df()
    plot_seizure_count_histogram(seizures_index_df)
    plt.show()
