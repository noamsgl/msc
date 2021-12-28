import pickle

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_validate

from msc.config import get_config
from msc.dataset import PSPDataset
from msc.dataset.dataset import get_datasets_df, MaskedDataset


def plot_feature_selection_mask(selector, patient_name, feature_name, classifier_name, ax):
    ax.set_title(f"RFE Support for {classifier_name} Classifier\n{feature_name}, {patient_name} Dataset")
    im = ax.imshow(selector.support_.reshape(-1, 60))
    ax.set_xlabel('time (5 s frames)')
    ax.set_ylabel('index of channel pair')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    return None


def plot_effect_of_feature_selection(score_before, score_after, ax):
    ax.set_title('Effect of feature selection on ROC AUC')
    scores = [score_before, score_after]
    ax.bar(["before", "after"], scores)
    ax.text(-0.125, scores[0] + 0.00125, f'{scores[0]:2f}', color='blue', fontweight='bold')
    ax.text(-0.125 + 1, scores[1] + 0.00125, f'{scores[1]:2f}', color='blue', fontweight='bold')
    plt.tight_layout()
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
    selected_classifier_name = "Random Forest"  # make sure newly trained classifier matches
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

    print(f"plotting the mask")
    fig, ax = plt.subplots(1, figsize=(3, 9))
    plot_feature_selection_mask(selector, selected_patient, selected_feature, selected_classifier_name, ax=ax)
    plt.show()

    rfe_mask = selector.get_support()
    masked_dataset = MaskedDataset(data_dir, rfe_mask)
    masked_X, labels = masked_dataset.get_masked_X(), masked_dataset.get_labels()

    print("training a classifier on the masked dataset")
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    num_folds = 5
    scoring = ['precision', 'recall', 'roc_auc']

    # X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        masked_X, labels, test_size=0.4, random_state=42
    )
    clf.fit(X_train, y_train)

    print("scoring the newly trained classifier")
    cv_results = cross_validate(clf, X_test, y_test, cv=num_folds, scoring=scoring, return_estimator=True)
    cv_results_df = pd.DataFrame(cv_results)
    score_after = cv_results_df.mean().test_roc_auc

    print("plotting the scores before and after masking")
    fig, ax = plt.subplots(1, figsize=(4, 4))
    score_before = results_row.loc[:, 'test_roc_auc'].item()
    plot_effect_of_feature_selection(score_before, score_after, ax=ax)
    plt.show()
