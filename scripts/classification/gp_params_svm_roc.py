import seaborn as sns
import sklearn
import pandas as pd
from clearml import Task
from mlxtend.plotting import plot_decision_regions
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split

from msc.results_collectors import GPResultsCollector

if __name__ == '__main__':
    seed_everything(42)
    task = Task.init(project_name="density_estimation", task_name="gp_matern_params_svm", reuse_last_task_id=True)
    # hparams = {'num_samples': 16}
    # task.set_parameters(hparams)


    requested_params = ['covar_module.raw_outputscale', 'covar_module.base_kernel.raw_lengthscale']

    # get results_df
    RELOAD = False  # change this to True in order to download updated results from ClearML servers
    if RELOAD:
        results_df = GPResultsCollector(requested_params).results_df
    else:
        results_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\results\params\results_gp_dog1_params.csv"
        results_df = pd.read_csv(results_fpath)


    # plot data join plot (before inference)
    sns.jointplot(data=results_df,
                  x=requested_params[0], y=requested_params[1],
                  hue="label_desc", palette="muted", legend=False)
    plt.suptitle("Matern Kernel Params for Dog_1 Dataset")
    plt.show()

    # encode labels
    le = preprocessing.LabelEncoder()
    le.fit(results_df.label_desc)
    results_df["label"] = le.transform(results_df.label_desc)

    # get all data
    X = results_df[requested_params].to_numpy()
    y = results_df["label"].to_numpy()

    # split train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # train a classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    # plot decision regions
    plot_decision_regions(X, y, clf=clf, legend=2)

    # add axes annotations
    plt.xlabel(requested_params[0])
    plt.ylabel(requested_params[1])
    plt.title('SVM on Dog 1 GP Matern Params')

    # add legend
    L = plt.legend()
    L.get_texts()[0].set_text(le.classes_[0])
    L.get_texts()[1].set_text(le.classes_[1])
    plt.show()

    # plot ROC curve
    sklearn.metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
    plt.title("ROC curve for SVM on Dog 1 GP Matern Params")
    plt.show()