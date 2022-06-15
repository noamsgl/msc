"""
Noam Siegel
31 Jan 2022
Plot seizure occurrences on a timeline for dog database
"""
import matplotlib.pyplot as plt

from msc.canine_db_utils import get_onsets
from msc.plot_utils import plot_seizure_occurrences_timeline

if __name__ == '__main__':
    dog_num = 3
    onsets = get_onsets(dog_num=dog_num)
    plot_seizure_occurrences_timeline(onsets, f"Dog {dog_num}")
    plt.show()
