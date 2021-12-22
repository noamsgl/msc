import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from msc.dataset.dataset import get_datasets_df, PSPDataset


def plot_pattern_window(x: ndarray, feature_name: str, patient_name: str, window_name: str, label_desc: str, ax=None):
    ax.set_title(
        f"5 min " + r"$\bf{" + f"{label_desc}" + "}$" + f" pattern\nfor {patient_name}, {feature_name}, {window_name}")
    im = ax.imshow(x.reshape(-1, 60))
    ax.set_xlabel('time (5 s frames)')
    ax.set_ylabel('index of channel pair')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return None


def plot_preictal_and_interictal_windows(preictal_row: DataFrame, interictal_row: DataFrame, feature_name: str,
                                         patient_name: str):
    """
    Saves to disk a plot of a single feature pattern
    Args:
        X:
        patient_name:
        feature_name:
        window_name:
        output_path:

    Returns:

    """
    assert_frame_equal(
        preictal_row.loc[:, ['package', 'patient']].reset_index(drop=True),
        interictal_row.loc[:, ['package', 'patient']].reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    fig.suptitle(f"Class Comparison")

    plot_pattern_window(preictal_row.loc[:, "x"].item(), feature_name, patient_name,
                        f"W_{preictal_row.loc[:, 'window_id'].item()}", preictal_row.loc[:, "label_desc"].item(),
                        ax=axes[0])
    plot_pattern_window(interictal_row.loc[:, "x"].item(), feature_name, patient_name,
                        f"W_{interictal_row.loc[:, 'window_id'].item()}", interictal_row.loc[:, "label_desc"].item(),
                        ax=axes[1])
    return None


if __name__ == '__main__':
    datasets_df = get_datasets_df()
    selected_patient = "pat_3500"
    selected_feature = "time_corr"
    fold_num = 0
    data_dir: str = datasets_df.loc[(datasets_df['feature_name'] == selected_feature) & (
            datasets_df['patient_name'] == selected_patient), "data_dir"].item()
    samples_df = PSPDataset(data_dir).samples_df

    first_preictal = samples_df.loc[samples_df['label_desc'] == 'preictal'].head(1)
    first_interictal = samples_df.loc[samples_df['label_desc'] == 'interictal'].head(1)

    plot_preictal_and_interictal_windows(first_preictal, first_interictal, selected_feature, selected_patient)
    plt.tight_layout()
    plt.show()
