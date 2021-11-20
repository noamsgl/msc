import gc
import torch
import numpy as np
from gpytorch.distributions import MultivariateNormal

import msc
import gpytorch

MVN = MultivariateNormal


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert (isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def predictive_accuracy(model, likelihood, test_x, test_y):
    with torch.no_grad():
        predictive_posterior = likelihood(model(test_x))
        predictive_accuracy = predictive_posterior.log_prob(test_y)
        return predictive_accuracy.detach().item()

def train():
    raise NotImplementedError()


if __name__ == '__main__':
    """
    H: History trained on (seconds)
    F: Future forecast (seconds)
    L: Length of time to iterate over (seconds)
    dt: timestep for loss (seconds)
    offset: offset from beginning of file (seconds)
    
    """

    H = 0.5 * 60
    F = 0.5 * 60
    L = 0.5 * 60
    dt = 0.1  # in seconds
    offset = 0

    accuracies = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    for dataset in msc.data.datasets(H, F, L, dt, offset, device=device):
        train_x, train_y, test_x, test_y = dataset

        # instantiate likelihood & model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-2),
            noise_prior=gpytorch.priors.NormalPrior(0, 1)).to(device=device)
        model = ExactGPModel(train_x, train_y, likelihood).to(device=device)

        # train model
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[40])

        num_iters = 50

        for i in range(num_iters):
            # print(f"beginning {i=} with {torch.cuda.memory_summary(abbreviated=True)}")
            optimizer.zero_grad()
            output: MVN = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 10 == 0:
                print(f'Iteration {i} - loss = {loss.item():.2f} - noise = {model.likelihood.noise.item():e}')
        # clear cuda cache

        # print alive tensors and variables
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    del obj
            except:
                pass
        torch.cuda.empty_cache()
        gc.collect()

        # evaluate
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            accuracy = predictive_accuracy(model, likelihood, test_x, test_y)
            print(f"{accuracy=}")
            accuracies.append(accuracy)
    print(accuracies)
