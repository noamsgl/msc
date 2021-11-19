from typing import NewType, List

import torch
import numpy as np
from gpytorch.distributions import MultivariateNormal

import msc
import gpytorch

MVN = MultivariateNormal


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def predictive_accuracy(model, test_x, test_y):
    predictive_posterior = model.likelihood(model(test_x))
    predictive_accuracy = predictive_posterior.log_prob(test_y)
    return predictive_accuracy.detach().item()


if __name__ == '__main__':
    """
    H: History trained on: 5 minutes
    F: Future forecast: 1 minute
    L: Length of time to iterate over: 1 minute
    dt: timestep for loss: 0.1 seconds
    offset: offset from beginning of file
    
    """

    H = 5 * 60
    F = 1 * 60
    L = 1 * 60
    dt = 0.1  # in seconds
    offset = 0

    accuracies = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset in msc.data.datasets(H, F, L, dt, offset, device=device):
        train_x, train_y, test_x, test_y = dataset

        # instantiate likelihood & model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-2),
            noise_prior=gpytorch.priors.NormalPrior(0, 1)).to(device=device)
        model = ExactGPModel(train_x, train_y, likelihood).to(device=device)

        # train model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[40])

        num_iters = 50

        for i in range(num_iters):
            optimizer.zero_grad()
            output: MVN = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Iteration {i} - loss = {loss:.2f} - noise = {model.likelihood.noise.item():e}')
            scheduler.step()

        # evaluate
        model.eval()
        accuracy = predictive_accuracy(model, test_x, test_y)
        print(f"{accuracy=}")
        accuracies.append(accuracy)

print(accuracies)


