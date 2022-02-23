import datetime
import os
from typing import List

import gpytorch
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from msc import config
from msc.canine_db_utils import get_onsets, get_record_start
from msc.dataset import SingleSampleDataset
from msc.models import InteractingPointProcessGPModel
from msc.plot_utils import plot_seizure_occurrences_timeline, plot_seizure_intervals_histogram


class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


if __name__ == '__main__':
    # override parameters with provided dictionary
    hparams = {'random_seed': 42,
               'learning_rate': 1e-2,
               'n_epochs': 800,
               'time_step': 3600 * 2 * 2,
               'version': '0.1.0',
               'enable_progress_bar': True,
               'fast_dev_run': False,
               'graphic_verbose': True}

    logger_dirpath = f"{config['PATH'][config['RESULTS_MACHINE']]['LIGHTNING_LOGS']}/prior/ipp"
    imsave_dirpath = f"{logger_dirpath}/ipp_prior/{hparams['version']}/figures"
    os.makedirs(imsave_dirpath, exist_ok=True)

    seed_everything(hparams['random_seed'])

    dog_num = 3
    onset_datetimes: List[datetime.datetime] = get_onsets(dog_num)

    if hparams['graphic_verbose']:
        plot_seizure_occurrences_timeline(onset_datetimes, f"Dog 3")
        plt.savefig(f"{imsave_dirpath}/seizures_timeline.png")
        plt.clf()
        plot_seizure_intervals_histogram(onset_datetimes, "Dog 3")
        plt.savefig(f"{imsave_dirpath}/intervals_histogram.png")
        plt.clf()

    record_start = get_record_start(dog_num)

    onsets_minutes = torch.Tensor(
        [(onset - record_start).total_seconds() / hparams['time_step'] for onset in onset_datetimes])
    train_x = torch.arange(0, max(onsets_minutes))
    train_y = torch.zeros_like(train_x).index_fill_(0, onsets_minutes.long(), 1)
    train_dataloader = DataLoader(SingleSampleDataset(train_x, train_y), num_workers=0)

    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    model = InteractingPointProcessGPModel(hparams, train_x, train_y)

    # plot prior samples
    if hparams['graphic_verbose']:
        prior_distribution = likelihood(model(train_x))

        n_draws = 8
        for i in range(n_draws):
            plt.plot(np.array(train_x[:100]), np.array((prior_distribution.sample() + 2 * i)[:100]))
        plt.xlabel(f"time [{hparams['time_step']} * sec]")
        plt.title('Prior')
        plt.savefig(f"{imsave_dirpath}/prior.png")
        plt.clf()

    # define trainer and fit model
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        # dirpath=logger_dirpath,
        filename="gp_ipp-{epoch:03d}-{train_loss:.2f}",
        save_top_k=1,
        mode="min",
        every_n_epochs=hparams['n_epochs']
    )

    early_stop_callback = MyEarlyStopping(monitor="train_loss", min_delta=0.00, patience=5, verbose=False, mode="min")

    logger = CSVLogger(save_dir=logger_dirpath, name="ipp_prior", version=hparams['version'])

    trainer = Trainer(max_epochs=hparams['n_epochs'], log_every_n_steps=1, gpus=1, profiler=None,
                      callbacks=[early_stop_callback], fast_dev_run=hparams['fast_dev_run'],
                      logger=logger,
                      deterministic=True, enable_progress_bar=hparams['enable_progress_bar'])

    trainer.fit(model, train_dataloader)

    trainer.save_checkpoint(f"{logger_dirpath}/ipp_prior/{hparams['version']}/checkpoints/gp_ipp.ckpt")

    torch.save(model.gpmodel.state_dict(), f"{logger_dirpath}/ipp_prior/{hparams['version']}/checkpoints/gp_ipp_state_dict.pth")

    if hparams['graphic_verbose']:
        # plot posterior samples
        posterior_distribution = likelihood(model(train_x))
        n_draws = 8
        for i in range(n_draws):
            plt.plot(train_x[:100], (posterior_distribution.sample() + 2 * i)[:100])
        plt.xlabel(f"time [{hparams['time_step']} * sec]")
        plt.title('Posterior')
        plt.savefig(f"{imsave_dirpath}/posterior.png")
        plt.clf()

