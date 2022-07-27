import copy
import math
import os
import warnings
from functools import partial

import arviz as az
import beanmachine
import beanmachine.ppl as bm
import beanmachine.ppl.experimental.gp as bgp
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.gp.models import SimpleGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from IPython.display import Markdown

# Eliminate excess UserWarnings from Python.
# warnings.filterwarnings("ignore")

# Manual seed
torch.manual_seed(123)

# Other settings for the notebook.
smoke_test = "SANDCASTLE_NEXUS" in os.environ or "CI" in os.environ

# Tool versions
print("pytorch version: ", torch.__version__)
print("gpytorch version: ", gpytorch.__version__)

x_train = torch.linspace(0, 1, 11)
y_train = torch.sin(x_train * (2 * math.pi)) + torch.randn(x_train.shape) * 0.2
x_test = torch.linspace(0, 1, 51).unsqueeze(-1)

with torch.no_grad():
    plt.scatter(x_train.numpy(), y_train.numpy())
    plt.show()

class Regression(SimpleGP):
    def __init__(self, x_train, y_train, mean, kernel, likelihood, *args, **kwargs):
        super().__init__(x_train, y_train, mean, kernel, likelihood)

    def forward(self, data, batch_shape=()):
        """
        Computes the GP prior given data. This method should always
        return a `torch.distributions.MultivariateNormal`
        """
        shape = data.shape[len(batch_shape)]
        jitter = torch.eye(shape, shape) * 1e-5
        for _ in range(len(batch_shape)):
            jitter = jitter.unsqueeze(0)
        if isinstance(self.mean, gpytorch.means.Mean):
            # demo using gpytorch for MAP estimation
            mean = self.mean(data)
        else:
            # use Bean Machine for learning posteriors
            if self.training:
                mean = self.mean(batch_shape).expand(data.shape[len(batch_shape) :])
            else:
                mean = self.mean.expand(data.shape[:-1])  # overridden for evaluation
        cov = self.kernel(data) + jitter
        return MultivariateNormal(mean, cov)

kernel = gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.PeriodicKernel())
likelihood = gpytorch.likelihoods.GaussianLikelihood()
mean = gpytorch.means.ConstantMean()
gp = Regression(x_train, y_train, mean, kernel, likelihood)

optimizer = torch.optim.Adam(
    gp.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
gp.eval()  # this converts the BM model into a gpytorch model
num_iters = 1 if smoke_test else 100

for i in range(num_iters):
    optimizer.zero_grad()
    output = gp(x_train)
    loss = -mll(output, y_train)
    loss.backward()
    if i % 10 == 0:
        print(
            "Iter %d/%d - Loss: %.3f"
            % (
                i + 1,
                100,
                loss.item(),
            )
        )
    optimizer.step()


with torch.no_grad():
    observed_pred = likelihood(gp(x_test))
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x_train.numpy(), y_train.numpy(), "k*")
    # Plot predictive means as blue line
    ax.plot(x_test.squeeze().numpy(), observed_pred.mean.numpy(), "b")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_test.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-1, 1])
    ax.legend(["Observed Data", "Mean", "Confidence"])

