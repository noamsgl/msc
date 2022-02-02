import gpytorch
import pytorch_lightning as pl
import torch


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
    def __init__(self, train_x, train_y, likelihood):
        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
        num_tasks = train_y.size(-1)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(1.5)),
            num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        cover_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, cover_x)


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
            self.gpmodel = multiChannelEEGGPModel(train_x, train_y, self.likelihood)
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