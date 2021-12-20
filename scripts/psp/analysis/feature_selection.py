import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.feature_selection import RFE

from msc.config import get_config
from msc.dataset import PSPDataset
from msc.dataset.dataset import get_datasets_df


def plot_feature_selection_mask(selector, patient_name, feature_name, classifier_name, ax):
    ax.set_title(f"RFE Support for {classifier_name} Classifier\n{feature_name}, {patient_name} Dataset")
    im = ax.imshow(selector.support_.reshape(-1, 60))
    ax.set_xlabel('time (5 s frames)')
    ax.set_ylabel('index of channel pair')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.colorbar(im, cax=cax)
    ax.tight_layout()
    return None


if __name__ == '__main__':
    # load config
    config = get_config()
    # load results
    results_fpath = f"{config['PATH']['LOCAL']['RESULTS']}/complete_results.pkl"
    results = pickle.load(open(results_fpath, 'rb'))

    # select patient and features set
    selected_patient = "pat_3500"
    selected_feature = "max_cross_corr"
    fold_num = 0
    selected_classifier_name = "Random Forest"
    results_row = results.loc[selected_patient, selected_feature, selected_classifier_name].head(1)
    estimator = results_row.loc[:, 'estimator'].item()

    selector = RFE(estimator, n_features_to_select=60 * 10)

    datasets_df = get_datasets_df()
    data_dir = datasets_df.query(
        f"feature_name == '{selected_feature}' and patient_name == '{selected_patient}'").data_dir.item()
    dataset = PSPDataset(dataset_dir=data_dir)
    X, labels = dataset.get_X(), dataset.get_labels()

    print(f"fitting selector to data. please wait...")
    selector = selector.fit(X, labels)
    print(f"done fitting selector to data.")

    fig, ax = plt.subplots(1, figsize=(3, 9))
    plot_feature_selection_mask(selector, selected_patient, selected_feature, selected_classifier_name, ax=ax)
    plt.show()
