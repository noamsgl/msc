import matplotlib.pyplot as plt

from msc.dataset.dataset import get_data_index_df


def plot_patient_recording_length_histogram(data_index_df, ax=None):
    if ax is None:
        ax = plt.gca()
    data_index_df.groupby('patient').apply(lambda row: row.end_date.max() - row.meas_date.min()).astype(
        'timedelta64[h]').plot(kind='hist',
                               grid=True,
                               title='Distribution of Recording Length per Patient',
                               ax=ax)
    ax.set_xlabel('Recording Length per Patient (hours)')


if __name__ == '__main__':
    data_index_df = get_data_index_df()
    plot_patient_recording_length_histogram(data_index_df)
    plt.show()
