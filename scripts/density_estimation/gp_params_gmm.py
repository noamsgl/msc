"""
Noam Siegel

Estimate density with GMM of GP parameter results
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import seaborn as sns
import torch
from clearml import Task
from matplotlib.patches import Ellipse
from pyro.infer import NUTS, MCMC
from pytorch_lightning import seed_everything
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from msc.results_collectors import GPResultsCollector

if __name__ == '__main__':
    seed_everything(42)
    task = Task.init(project_name="density_estimation", task_name="gp_matern_params_gmm", reuse_last_task_id=False)
    hparams = {'num_samples': 100}
    task.set_parameters(hparams)

    requested_params = ['covar_module.raw_outputscale', 'covar_module.base_kernel.raw_lengthscale']
    RELOAD = False
    if RELOAD:
        results_df = GPResultsCollector(requested_params).results_df
    else:
        results_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\results\params\results_gp_dog1_params.csv"
        results_df = pd.read_csv(results_fpath)

    # plot joint plot
    sns.jointplot(data=results_df,
                  x=requested_params[0], y=requested_params[1],
                  hue="label_desc", palette="muted", legend=False)
    plt.suptitle("Matern Kernel Params for Dog_1 Dataset")
    plt.show()

    def is_pos_def(x):
        x = x.numpy()
        return np.all(np.linalg.eigvals(x) > 0)


    def gmm_separate_scales(data: Tensor, K: int = 2):
        # K is number of mixture components
        # D is dimension of each data point
        D = data.size(-1)

        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
        with pyro.plate('components', K):
            locs = pyro.sample('locs', dist.Normal(torch.zeros(D), 10 * torch.ones(D)).to_event(1))
            M = pyro.sample('matrix', dist.Normal(torch.zeros((D, D)), 1 * torch.ones((D, D))).to_event(2))
            M_transposed = torch.transpose(M, -1, -2)
            scales = torch.bmm(M, M_transposed)
        with pyro.plate('data', len(data)):
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            obs = pyro.sample('obs', dist.MultivariateNormal(locs[assignment], scales[assignment]), obs=data)
        return


    def run_mcmc(model, args, num_samples):
        """initialize and perform mcmc.run() with NUTS kernel"""
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples)
        mcmc.run(*args)
        return mcmc


    def preprocess_data(X):
        """normalize"""
        sclr = StandardScaler()
        X = sclr.fit_transform(X)
        return torch.tensor(X).float()

    # preprocess data
    X = torch.Tensor(results_df[requested_params].to_numpy())

    X = preprocess_data(X)

    # sample
    mcmc_separate = run_mcmc(gmm_separate_scales, (X,), num_samples=hparams['num_samples'])
    chn_separate = mcmc_separate.get_samples()
    mcmc_separate.summary()


    def visualize_gmm_separate_posterior():
        chain = chn_separate
        locs = chain["locs"].mean(dim=0)
        M = chain["matrix"].mean(dim=0)
        M_transposed = torch.transpose(M, -1, -2)
        scales = torch.bmm(M, M_transposed)
        mus = locs
        sigmas = scales

        # get figure
        fig = plt.figure()

        # plot data
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=results_df["label_desc"], palette="muted")

        # plot GMM centroids
        x = [float(m[0]) for m in mus.data]
        y = [float(m[1]) for m in mus.data]
        plt.scatter(x, y, 99, c='red')

        # plot ellipses for each cluster
        for sig_ix in range(2):
            ax = fig.gca()
            cov = sigmas[sig_ix]
            lam, v = np.linalg.eig(cov)
            lam = np.sqrt(lam)
            ell = Ellipse(xy=(x[sig_ix], y[sig_ix]),
                          width=lam[0] * 4,
                          height=lam[1] * 4,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          color='blue')
            ell.set_facecolor('none')
            ax.add_artist(ell)

        # show
        return fig

    visualize_gmm_separate_posterior()
    plt.show()

