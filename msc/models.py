import gpytorch
import pytorch_lightning as pl
import torch
from gpytorch.variational import CholeskyVariationalDistribution, UnwhitenedVariationalStrategy, VariationalStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR


class EEGGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        cover_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cover_x)


class multiChannelEEGGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_channels: int = 1):
        assert num_channels == train_y.size(-1), f"error: {num_channels=} does not match {train_y.size(-1)=}"
        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
        num_tasks = train_y.size(-1)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(1.5),
            num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        cover_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, cover_x)

    def get_params_dict(self):
        params_dict = {}
        for param_name, param in self.named_parameters():
            if param.numel() == 1:
                params_dict[param_name] = param.item()
            else:
                for i in range(param.numel()):
                    params_dict[f"{param_name}[{i}]"] = param[i].item()
        return params_dict


class SingleSampleEEGGPModel(pl.LightningModule):
    def __init__(self, hparams, train_x, train_y):
        super().__init__()
        self.save_hyperparameters(hparams)

        if train_y.dim() == 1:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gpmodel = EEGGPModel(train_x, train_y, self.likelihood)
        elif train_y.dim() == 2:
            num_tasks = train_y.size(-1)
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            self.gpmodel = multiChannelEEGGPModel(train_x, train_y, self.likelihood, num_channels=num_tasks)
        else:
            raise ValueError(f"train_y dimension is {train_y.dim()}, should be 1 or 2")
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gpmodel)

    def forward(self, x):
        # compute prediction
        return self.gpmodel(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        assert batch[0].shape[0] == 1 and batch[1].shape[0] == 1, "error: batch size must be 1"
        x, y = batch
        x, y = x.squeeze(), y.squeeze()
        # pass through the forward function
        output = self(x)

        # calc loss
        loss = -self.mll(output, y)
        self.log('train_loss', loss)

        # log model parameters
        for param_name, param in self.gpmodel.named_parameters():
            if param.numel() == 1:
                self.log(param_name, param.item())
            else:
                for i in range(param.numel()):
                    self.log(f"{param_name}[{i}]", param[i].item())

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()


class HawkesProcessGP(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, likelihood, inducing_points, num_cycles):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        super(HawkesProcessGP, self).__init__(variational_strategy)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.RBFKernel() * gpytorch.kernels.CosineKernel() + gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.MaternKernel(3 / 2))
        if num_cycles == 0:
            self.covar_module = gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.MaternKernel(3/2)
        elif num_cycles == 1:
            self.covar_module = gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel()
        elif num_cycles == 2:
            self.covar_module = gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel()
        elif num_cycles == 3:
            self.covar_module = gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel()
        else:
            raise ValueError("only supporting up to 3 cycles")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class InteractingPointProcessGPModel(pl.LightningModule):
    def __init__(self, hparams, train_x, train_y, inducing_points):
        super().__init__()
        self.save_hyperparameters(hparams)

        likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.gpmodel = HawkesProcessGP(train_x, likelihood, inducing_points, hparams['num_cycles'])

        # "Loss" for GPs - marginal log likelihood
        self.mll = gpytorch.mlls.VariationalELBO(likelihood, self.gpmodel, train_y.numel())

    def forward(self, x):
        # compute prediction
        return self.gpmodel(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.squeeze(), y.squeeze(),
        # pass through the forward function
        output = self(x)

        # calc loss
        loss = -self.mll(output, y)
        self.log('train_loss', loss)

        # log model parameters
        for param_name, param in self.gpmodel.named_parameters():
            if 'variational_strategy' not in param_name:
                if param.numel() == 1:
                    self.log(param_name, param.item())
                else:
                    for i in range(param.numel()):
                        self.log(f"{param_name}[{i}]", param[i].item())

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()
