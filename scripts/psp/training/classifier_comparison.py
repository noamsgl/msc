from datetime import datetime

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from msc.dataset.dataset import PSPDataset

# initialize save param
save_to_disk = False
# load data
data_dir = r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\max_cross_corr\surfCO\pat_3500\20211213T182128"

dataset = PSPDataset(data_dir)
X, labels = dataset.get_X(), dataset.get_labels()

# initialize classifier names
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

# initialize datasets
datasets = [
    (X, labels),
]

# iterate over datasets
num_folds = 5
results = pd.DataFrame()

for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    scoring = ['precision', 'recall', 'roc_auc']
    score_cols = [f'test_{sc}' for sc in scoring]
    # iterate over classifiers
    for name, clf in tqdm(list(zip(names, classifiers)), desc="iterating over classifiers"):
        clf.fit(X_train, y_train)
        cv_results = cross_validate(clf, X_test, y_test, cv=num_folds, scoring=scoring, return_estimator=True)
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df["name"] = name
        cv_results_df["ds_cnt"] = ds_cnt
        results = results.append(cv_results_df)

results["fold"] = results.index
results = results.set_index('name')

print(f'{results=}')

if save_to_disk:
    results.to_csv(f'results_{datetime.now()}.csv')

