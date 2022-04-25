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
from torch.utils.data import DataLoader, TensorDataset

from msc import config
from msc.canine_db_utils import get_onsets, get_record_start
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
               'patience': 6,
               'batch_size': 2048,
               'inducing_time_step': 60 * 60 * 4,  # in seconds
               'cholesky_jitter_double': 1e-1,  # default None
               'real_time_step': 60 * 10,  # in seconds
               'num_cycles': 0,
               'version': '0.1.7',
               'enable_progress_bar': True,
               'fast_dev_run': False,
               'graphic_verbose': True}

    # set gpytorch settings
    with gpytorch.settings.cholesky_jitter(float=None, double=hparams['cholesky_jitter_double'], half=None):

        # set logger paths
        logger_dirpath = f"{config['PATH'][config['RESULTS_MACHINE']]['LIGHTNING_LOGS']}/prior/ipp"
        imsave_dirpath = f"{logger_dirpath}/ipp_prior/{hparams['version']}/figures"
        os.makedirs(imsave_dirpath, exist_ok=True)

        # set rng seed
        seed_everything(hparams['random_seed'])

        # get data
        dog_num = 3
        onset_datetimes: List[datetime.datetime] = get_onsets(dog_num)
        record_start = get_record_start(dog_num)

        # plot data
        if hparams['graphic_verbose']:
            plot_seizure_occurrences_timeline(onset_datetimes, f"Dog 3")
            plt.savefig(f"{imsave_dirpath}/seizures_timeline.png")
            plt.clf()
            plot_seizure_intervals_histogram(onset_datetimes, "Dog 3")
            plt.savefig(f"{imsave_dirpath}/intervals_histogram.png")
            plt.clf()

        # instantiate training set
        onsets_real = torch.Tensor(
            [(onset - record_start).total_seconds() / hparams['real_time_step'] for onset in onset_datetimes])
        train_x = torch.arange(0, max(onsets_real))
        train_y = torch.zeros_like(train_x).index_fill_(0, onsets_real.long(), 1)
        train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=hparams['batch_size'], shuffle=True,
                                      num_workers=0)

        # instantiate inducing points
        onsets_inducing = torch.Tensor(
            [(onset - record_start).total_seconds() / hparams['inducing_time_step'] for onset in onset_datetimes])
        inducing_points = torch.arange(0, max(onsets_inducing))

        likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        model = InteractingPointProcessGPModel(hparams, train_x, train_y, inducing_points)

        # instantiate test set
        test_x = train_x[:200]

        # plot prior samples
        if hparams['graphic_verbose']:
            prior_distribution = likelihood(model(test_x))
            n_draws = 8
            for i in range(n_draws):
                plt.plot(np.array(test_x), np.array((prior_distribution.sample() + 2 * i)))
            plt.xlabel(f"time [{hparams['real_time_step']} * sec]")
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

        # instantiate early stopping
        early_stop_callback = MyEarlyStopping(monitor="train_loss", min_delta=0.00, patience=hparams['patience'],
                                              verbose=False, mode="min")

        # instantiate logger
        logger = CSVLogger(save_dir=logger_dirpath, name="ipp_prior", version=hparams['version'])

        # instantiate trainer and fit model
        trainer = Trainer(max_epochs=hparams['n_epochs'], log_every_n_steps=1, gpus=1, profiler=None,
                          callbacks=[early_stop_callback], fast_dev_run=hparams['fast_dev_run'],
                          logger=logger,
                          deterministic=True, enable_progress_bar=hparams['enable_progress_bar'])

        trainer.fit(model, train_dataloader)

        print(f"finished model fit")
        # save checkpoint and state_dict
        trainer.save_checkpoint(f"{logger_dirpath}/ipp_prior/{hparams['version']}/checkpoints/gp_ipp.ckpt")

        torch.save(model.gpmodel.state_dict(),
                   f"{logger_dirpath}/ipp_prior/{hparams['version']}/checkpoints/gp_ipp_state_dict.pth")

        # plot posterior samples
        if hparams['graphic_verbose']:
            posterior_distribution = likelihood(model(test_x))
            n_draws = 8
            for i in range(n_draws):
                plt.plot(test_x, (posterior_distribution.sample() + 2 * i))
            plt.xlabel(f"time [{hparams['real_time_step']} * sec]")
            plt.title('Posterior')
            plt.savefig(f"{imsave_dirpath}/posterior.png")
            plt.clf()
