import os
import gpytorch
import hydra
import logging
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

from msc.datamodules.data_utils import SingleSampleDataset


class GPEmbeddor:
    def __init__(self, random_seed, num_channels, learning_rate, n_epochs, fast_dev_run, resume_from_checkpoint=False) -> None:
        self.random_seed = random_seed
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.fast_dev_run = fast_dev_run
        # init GP hyperparameters
        self.hparams = {'random_seed': random_seed,
                        'num_channels': num_channels,
                        'learning_rate': learning_rate,
                        'n_epochs': n_epochs,
                        'version': '0.2.0',
                        'enable_progress_bar': False,
                        'fast_dev_run': fast_dev_run,
                        'graphic_verbose': False}
        # convert relative ckpt path to absolute path if necessary
        ckpt_path = resume_from_checkpoint
        if resume_from_checkpoint and not os.path.isabs(ckpt_path):
            resume_from_checkpoint = os.path.join(
                hydra.utils.get_original_cwd(), ckpt_path
            )


    def embed(self, data, logger_dirpath):        
        # Set seed for random number generators in pytorch, numpy and python.random
        seed_everything(self.random_seed, workers=True)

        # init data
        train_x = torch.linspace(0, 1, len(data))
        train_y = torch.Tensor(data)
        train_dataloader = DataLoader(SingleSampleDataset(train_x, train_y), num_workers=0)  # type: ignore

        # init GP model
        model = SingleSampleEEGGPModel(self.hparams, train_x, train_y)

        # define trainer and fit model
        checkpoint_callback = ModelCheckpoint(
                monitor="train_loss",
                # dirpath=logger_dirpath,
                filename="gp_inference-{epoch:03d}-{train_loss:.2f}",
                save_top_k=1,
                mode="min",
                every_n_epochs=self.hparams['n_epochs']
        )
        
        logger = CSVLogger(save_dir=logger_dirpath, name="gp_embedding", version=self.hparams['version'])

        
        trainer = Trainer(max_epochs=self.hparams['n_epochs'], log_every_n_steps=200, gpus=1, profiler=None,
                            callbacks=[checkpoint_callback], fast_dev_run=False, logger=logger,
                            deterministic=True, enable_progress_bar=self.hparams['enable_progress_bar'])

        trainer.fit(model, train_dataloader)
        
        # TODO:collect results as embedding

        # save low dimensional vector to file
        print(data.shape)
        return data.flatten()



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
