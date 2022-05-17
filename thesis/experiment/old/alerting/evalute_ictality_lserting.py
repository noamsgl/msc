"""
evaluate_ictality_alerting.py
Noam Siegel
16 Feb 2022

1) plot train and test embeddings together
2) provide p-values for test set
3) provide

"""

# imports
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning import seed_everything
from sklearn import mixture
from sklearn.manifold import TSNE

from msc.results_collectors import GPResultsCollector

seed_everything(42)

if __name__ == '__main__':
    # load embeddings

    # get results of multitask (pair) GP params MLE
    requested_project_name = "inference/pairs"
    requested_params = ['covar_module.data_covar_module.base_kernel.raw_lengthscale',
                        'covar_module.task_covar_module.covar_factor[0]',
                        'covar_module.task_covar_module.covar_factor[1]',
                        'covar_module.task_covar_module.raw_var[0]',
                        'covar_module.task_covar_module.raw_var[1]']

    results_df = GPResultsCollector.from_clearml(requested_project_name, requested_params, n_pages_limit=1).results_df

    # get parameter values
    X = results_df[requested_params].values

    # calculate t-SNE values of parameters
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)

    # add t-SNE results to results_df
    results_df['tsne-2d-one'] = tsne_results[:, 0]
    results_df['tsne-2d-two'] = tsne_results[:, 1]

    sns.scatterplot(data=results_df,
                    x='tsne-2d-one', y='tsne-2d-two',
                    # hue_order=hue_order,
                    hue="label_desc", palette="muted", legend=True)
    plt.suptitle("2-channels Multitask Matern Params for Dog_1 Dataset")
    plt.show()

    # estimate interictal pdf
    # get parameter values of interictal results
    X = results_df.loc[results_df['label_desc'] == 'interictal', requested_params]

    # fit a Gaussian Mixture Model with two components
    gmm = mixture.GaussianMixture(n_components=2, covariance_type="full")
    gmm.fit(X)
