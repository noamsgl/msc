"""
Noam Siegel

Estimate density with GMM of GP parameter results
Report log likelihood of observations

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from clearml import Task
from joblib import dump
from matplotlib.colors import LogNorm
from pytorch_lightning import seed_everything
from sklearn import mixture
from sklearn.preprocessing import StandardScaler

from msc import config
from msc.data_utils import get_time_as_str
from msc.results_collectors import GPResultsCollector


def standard_scale_data(X):
    """normalize"""
    sclr = StandardScaler()
    X = sclr.fit_transform(X)
    return torch.tensor(X).float()


if __name__ == '__main__':
    seed_everything(42)
    task = Task.init(project_name="density_estimation", task_name="gp_matern_params_gmm_sklearn",
                     reuse_last_task_id=True)
    hparams = {'num_samples': 50,
               'selected_label_desc': "ictal"}
    task.set_parameters(hparams)

    requested_params = ['covar_module.raw_outputscale', 'covar_module.base_kernel.raw_lengthscale']

    # get results_df
    RELOAD = False  # change this to True in order to download updated results from ClearML servers
    if RELOAD:
        results_df = GPResultsCollector.from_clearml(requested_params).results_df
    else:
        results_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\results\params\results_gp_dog1_params.csv"
        results_df = pd.read_csv(results_fpath)

    # plot data join plot (selected label desc)
    hue_order = ["ictal", "interictal"]
    filtered_results_df = results_df.loc[results_df["label_desc"] == hparams['selected_label_desc']]
    sns.jointplot(data=filtered_results_df,
                  x=requested_params[0], y=requested_params[1],
                  # hue_order=hue_order,
                  hue="label_desc", palette="muted", legend=True)
    plt.suptitle("Matern Kernel Params for Dog_1 Dataset")
    plt.show()

    # selected label_desc parameters
    X = torch.Tensor(
        results_df.loc[results_df["label_desc"] == hparams['selected_label_desc'], requested_params].to_numpy())

    # standard scale parameters
    X = standard_scale_data(X)

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
    clf.fit(X)

    # persist GMM to disk
    print("dumping classifier with joblib")
    dump(clf,
         f"{config['PATH']['LOCAL']['RESULTS']}/density_estimation_models/GMM_{hparams['selected_label_desc']}_{get_time_as_str()}.joblib")

    # display predicted scores by the model as a contour plot
    x = np.linspace(-10.0, 10.0)
    y = np.linspace(-10.0, 10.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8, extend="both")
    plt.scatter(X[:, 0], X[:, 1], 0.8)

    # add axes annotations
    plt.xlabel(requested_params[0])
    plt.ylabel(requested_params[1])

    # add title
    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    plt.show()


    def compute_log_likelihood(row):
        XX = np.array(row[requested_params]).reshape(1, -1)
        return clf.score_samples(XX).item()


    # estimate probability densities of all results
    results_df["log_likelihood"] = results_df.apply(compute_log_likelihood, axis=1)

    # plot data probability density (after inference)
    sns.displot(data=results_df, x="log_likelihood", hue="label_desc", palette="muted", legend=True)
    plt.suptitle("GMM log likelihood for Dog_1 GP Matern Params")
    plt.show()

    task.close()
