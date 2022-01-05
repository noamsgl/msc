"""
Conditions a generative model to produce seizures of shape (n_channels, n_times)

Output a single, randomly generated seizure


"""
import gpytorch
import matplotlib.pyplot as plt
import torch

from msc import config
from msc.dataset.dataset import get_data_index_df, SeizuresDataset, get_seizures_index_df


class MultitaskGPSeizureModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPSeizureModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':

    # load data
    data_dir = None
    data_dir = r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\SEIZURES\20220103T101554"
    if data_dir is None:
        data_index_df = get_data_index_df()
        seizures_index_df = get_seizures_index_df()
        dataset = SeizuresDataset.generate_dataset(seizures_index_df,
                                                   time_minutes_before=0, time_minutes_after=0,
                                                   fast_dev_mode=True,
                                                   )
    else:
        dataset = SeizuresDataset(data_dir, preload_data=True)

    # set signal properties
    d = 2
    sfreq = 256  # todo: get from dataset
    crop_seconds = 400

    # get X (times), Y (samples)
    train_x = dataset.get_train_x(sfreq=sfreq, num_channels=d, crop=crop_seconds)
    train_y = dataset.get_train_y(sfreq, num_channels=d, crop=crop_seconds)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=d)
    model = MultitaskGPSeizureModel(train_x, train_y, likelihood)

    smoke_test = config['CI']
    training_iterations = 2 if smoke_test else 50

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    # make predictions

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # todo: Initialize plots for d != 2
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
