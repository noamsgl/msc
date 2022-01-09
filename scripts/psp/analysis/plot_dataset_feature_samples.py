from pandas import DataFrame
import matplotlib.pyplot as plt
from tqdm import tqdm

from msc.dataset import PSPDataset
from msc.dataset import get_datasets_df
from scripts.psp.analysis.plot_feature_window import plot_feature_window


def plot_dataset_feature_samples(selected_patient: str, patients_datasets_df: DataFrame):
    nrows = len(patients_datasets_df)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharex=True, sharey=False)
    for i in tqdm(range(nrows), desc="generating figures"):
        dataset = patients_datasets_df.loc[i]
        psp_dataset = PSPDataset(dataset.data_dir)
        X, labels = psp_dataset.get_X(), psp_dataset.get_labels(format='desc')
        for j in range(ncols):
            plot_feature_window(X[j].reshape(-1, 60), patient_name=selected_patient, window_name=f'W_{j}',
                                feature_name=dataset.feature_name, ax=axes[i, j])
    plt.tight_layout()
    return None

if __name__ == '__main__':
    selected_patient = 'pat_3700'
    datasets_df = get_datasets_df()
    patients_datasets_df = datasets_df.query(f"patient_name == '{selected_patient}'").reset_index()

    plot_dataset_feature_samples(selected_patient, patients_datasets_df)
    plt.show()