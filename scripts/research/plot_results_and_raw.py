import matplotlib.pyplot as plt
import pickle
import mne
from mne.io import BaseRaw
import numpy as np


def get_data(**kwargs):
    # raw_path = kwargs['raw_path']
    # resample_sfreq = kwargs['resample_sfreq']
    raw_path = r"C:\raw_data\surf30\pat_103002\adm_1030102\rec_103001102\103001102_0113.data"
    raw: BaseRaw = mne.io.read_raw_nicolet(raw_path, ch_type='eeg')
    raw = raw.pick(kwargs['picks'])
    resample_sfreq = 128  #todo: remove this
    raw = raw.resample(resample_sfreq)
    raw = raw.crop(tmin=0, tmax=200)
    data, times = raw.get_data(return_times=True)
    data = data.squeeze()
    data = (data - np.mean(data)) / np.std(data)
    return data, times


if __name__ == '__main__':

    results_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\output\run_20211122T131852.pkl"

    results = pickle.load(open(results_fpath, 'rb'))

    data, times = get_data(**results)

    result_times = np.array(results['times'])
    result_indices = np.nonzero(np.in1d(times, result_times))[0]
    result_data = data[result_indices]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('EEG (z-score)', color=color)
    ax1.plot(times, data, label='raw EEG (channel C3)', color=color)

    # ax1.tick_params

    ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis

    color = 'tab:red'
    ax2.semilogy(result_times, -np.array(results['accuracies']), label=r'$-\log(p(y|x))$', color=color)
    ax2.set_ylabel('predictive accuracy', color=color)

    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.title('single channel EEG and predictive accuracy\n history=60s, future=30s')
    # fig.tight_layout()
    plt.show()

