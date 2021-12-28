import pandas as pd
from pandas import DataFrame
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from msc.config import get_config
from msc.data_utils.load import get_time_as_str
from msc.dataset.dataset import PSPDataset, get_datasets_df


def main(save_to_disk=True, feature_names=('max_cross_corr', 'phase_lock_val', 'spect_corr', 'time_corr'),
         patient_names=('pat_3500', 'pat_3700', 'pat_7200'),
         classifier_names=None) -> DataFrame:
    # get config
    config = get_config()

    # load data
    # initialize datasets
    feature_names = feature_names
    patient_names = patient_names

    datasets_df = get_datasets_df()
    datasets_df = datasets_df.loc[
        datasets_df['feature_name'].isin(feature_names) & datasets_df['patient_name'].isin(patient_names)]

    # initialize classifier names
    if classifier_names is None:
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            # "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]
    else:
        names = classifier_names

    # initialize classifiers
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    # iterate over datasets
    num_folds = 5
    scoring = ['precision', 'recall', 'roc_auc']
    score_cols = [f'test_{sc}' for sc in scoring]
    results = pd.DataFrame(columns=["dataset"])
    dataset_results_dfs = []
    for idx, ds in tqdm(list(datasets_df.iterrows()), desc="iterating datasets"):
        print(f"beginning {ds=}")
        # load data
        dataset = PSPDataset(ds['data_dir'])
        X, y = dataset.get_X(), dataset.get_labels()

        # preprocess dataset, split into training and test part
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        # iterate over classifiers
        cv_results_dfs = []
        for name, clf in tqdm(list(zip(names, classifiers)), desc="iterating over classifiers"):
            clf.fit(X_train, y_train)
            cv_results = cross_validate(clf, X_test, y_test, cv=num_folds, scoring=scoring, return_estimator=True)
            cv_results_df = pd.DataFrame(cv_results)

            # cross merge cv_results with dataset information
            cv_results_df["classifier_name"] = name
            cv_results_df = pd.merge(cv_results_df, ds.to_frame().transpose(), how='cross')
            cv_results_df = cv_results_df.rename_axis('fold').reset_index()
            cv_results_dfs.append(cv_results_df)

        dataset_results_df = pd.concat(cv_results_dfs)
        dataset_results_dfs.append(dataset_results_df)

    results = pd.concat(dataset_results_dfs)

    if save_to_disk:
        results.to_csv(f'results_{get_time_as_str()}.csv')

    return results


if __name__ == '__main__':
    main()
