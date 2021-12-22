import matplotlib.pyplot as plt
import numpy as np

from msc.dataset.dataset import get_datasets_df
from scripts.psp.analysis.plot_dataset_2d_pca_ import plot_pca_projection

plt.style.use('fivethirtyeight')
if __name__ == '__main__':
    datasets_df = get_datasets_df()
    n_unique_patients = len(datasets_df.loc[:, "patient_name"].unique())
    n_unique_features = len(datasets_df.loc[:, "feature_name"].unique())
    fig, axes = plt.subplots(n_unique_features, n_unique_patients,
                             figsize=(4.2 * n_unique_patients, 4.2 * n_unique_features), sharex=True, sharey=True)
    fig, axes = plt.subplots(n_unique_patients, n_unique_features,
                             figsize=(4.2 * n_unique_features, 4.2 * n_unique_patients), sharex=True, sharey=True)

    for i, ax in enumerate(np.array(fig.axes).T):
        plot_pca_projection(datasets_df.sort_values(by=['patient_name', 'feature_name'], ignore_index=True).loc[i],
                            ax=ax)
    plt.tight_layout()
    plt.show()
