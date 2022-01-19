import gpytorch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from clearml import Task
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from torch import Tensor
from torch.utils.data import DataLoader

from msc import config
from msc.dataset import DogDataset, SingleSampleDataset


class EEGGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        cover_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cover_x)


class SingleSampleEEGGPModel(pl.LightningModule):
    def __init__(self, hparams, train_x, train_y):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gpmodel = EEGGPModel(train_x, train_y, self.likelihood)

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
        for param_name, param in model.named_parameters():
            self.log(param_name, param.item())
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()


if __name__ == '__main__':
    seed_everything(42)
    n_draws = 8
    dataset_dir = r"C:\Users\noam\Repositories\noamsgl\msc\data\seizure-detection\Dog_1"
    dataset = DogDataset(dataset_dir)
    samples_df = dataset.samples_df

    selected_fname = 'Dog_1_ictal_segment_2.mat'

    selected_ch_names = ['NVC0905_22_002_Ecog_c001', 'NVC0905_22_002_Ecog_c002',
                         'NVC0905_22_002_Ecog_c003', 'NVC0905_22_002_Ecog_c004',
                         'NVC0905_22_002_Ecog_c005', 'NVC0905_22_002_Ecog_c006',
                         'NVC0905_22_002_Ecog_c007', 'NVC0905_22_002_Ecog_c008',
                         'NVC0905_22_002_Ecog_c009', 'NVC0905_22_002_Ecog_c010',
                         'NVC0905_22_002_Ecog_c011', 'NVC0905_22_002_Ecog_c012',
                         'NVC0905_22_002_Ecog_c013', 'NVC0905_22_002_Ecog_c014',
                         'NVC0905_22_002_Ecog_c015', 'NVC0905_22_002_Ecog_c016'
                         ]
    samples_df = samples_df.loc[:,
        # samples_df['fname'] == selected_fname,
        selected_ch_names + ['time', 'fname']]

    for fname, group in samples_df.groupby('fname'):
        for ch_name in selected_ch_names:
            print(f"beginning training with {fname}/{ch_name}")
            task = Task.init(project_name=f"inference/{fname}", task_name=f"gp_{ch_name}")
            # override parameters with provided dictionary
            hparams = {'learning_rate': 1e-2,
                       'n_epochs': 1000}
            task.set_parameters(hparams)

            train_x = Tensor(group['time'].values)
            train_y = Tensor(group[ch_name].values)
            train_dataloader = DataLoader(SingleSampleDataset(train_x, train_y), num_workers=0)
            plt.plot(train_x, train_y)
            plt.xlabel('time (s)')
            plt.title('Data Sample')
            plt.show()

            model = SingleSampleEEGGPModel(hparams, train_x, train_y)

            for i in range(n_draws):
                plt.plot(train_x, model(train_x).sample())
            plt.xlabel('time (s)')
            plt.title('Prior')
            plt.show()

            # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss",
                dirpath=f"{config['PATH']['LOCAL']['LIGHTNING_LOGS']}/{fname[:-4]}/{ch_name[-4:]}",
                filename="gp_inference-{epoch:03d}-{train_loss:.2f}",
                save_top_k=1,
                mode="min",
                every_n_epochs=1000
            )

            trainer = Trainer(max_epochs=hparams['n_epochs'], log_every_n_steps=1, gpus=1, profiler=False,
                              callbacks=[checkpoint_callback], fast_dev_run=False)
            trainer.fit(model, train_dataloader)
            # trainer.save_checkpoint('trained_model.ckpt')

            for i in range(n_draws):
                plt.plot(train_x, model(train_x).sample())
            plt.xlabel('time (s)')
            plt.title('Posterior')
            plt.show()

            # trainer.test(test_dataloaders=train_dataloader)

            task.close()
