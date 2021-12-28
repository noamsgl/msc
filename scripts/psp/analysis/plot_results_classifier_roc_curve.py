import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from msc.dataset.dataset import get_datasets_df, PSPDataset


def plot_classifier_roc_curve_and_confusion_matrix(selected_patient, selected_feature, selected_classifier):
    datasets_df = get_datasets_df()

    dataset = datasets_df.query(
        f"feature_name == '{selected_feature}' and patient_name == '{selected_patient}'")
    psp_dataset = PSPDataset(dataset.data_dir.item())

    results_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\results\complete_results.pkl"
    results = pickle.load(open(results_fpath, 'rb'))

    estimator = results.loc[
        selected_patient, selected_feature, selected_classifier, results['fold'] == 0].estimator.item()

    X, labels = psp_dataset.get_X(), psp_dataset.get_labels()
    le = LabelEncoder()
    le.fit(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, le.transform(labels), test_size=0.4, random_state=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Evaluation of {selected_classifier} on\n{selected_patient}, {selected_feature}')
    ConfusionMatrixDisplay.from_estimator(estimator, X_test, y_test, ax=axes[0])
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=axes[1])
    return None


def plot_classifier_roc_curve(selected_patient, selected_feature, selected_classifier):
    datasets_df = get_datasets_df()

    dataset = datasets_df.query(
        f"feature_name == '{selected_feature}' and patient_name == '{selected_patient}'")
    psp_dataset = PSPDataset(dataset.data_dir.item())

    results_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\results\complete_results.pkl"
    results = pickle.load(open(results_fpath, 'rb'))

    estimator = results.loc[
        selected_patient, selected_feature, selected_classifier, results['fold'] == 1].estimator.item()

    X, labels = psp_dataset.get_X(), psp_dataset.get_labels()
    le = LabelEncoder()
    le.fit(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, le.transform(labels), test_size=0.4, random_state=0)

    fig, ax = plt.subplots(1, figsize=(4, 4))
    ax.set_title(f'Evaluation of {selected_classifier} on\n{selected_patient}, {selected_feature}')
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    plt.tight_layout()
    return None


if __name__ == '__main__':
    selected_patient = 'pat_7200'
    selected_feature = 'time_corr'
    selected_classifier = 'Linear SVM'

    plot_classifier_roc_curve(selected_patient, selected_feature, selected_classifier)
    plt.show()
