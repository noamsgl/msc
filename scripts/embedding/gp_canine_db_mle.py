import sys

import matplotlib.pyplot as plt
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader

from msc import config
from msc.dataset import DogDataset, SingleSampleDataset
from msc.models import SingleSampleEEGGPModel

if __name__ == '__main__':
    # override parameters with provided dictionary
    hparams = {'random_seed': 42,
               'num_channels': 1,
               'selected_label_desc': 'test',
               'learning_rate': 1e-2,
               'n_epochs': 1000,
               'version': '0.1.0',
               'enable_progress_bar': False,
               'fast_dev_run': False}

    n_draws = 8  # number of samples to draw for plotting
    dataset_dir = f"{config['PATH'][config['RAW_MACHINE']]['DATA_DIR']}/seizure-detection/Dog_1"
    dataset = DogDataset(dataset_dir, include_test=True)
    samples_df = dataset.samples_df

    # define all channels
    all_ch_names = ['NVC0905_22_002_Ecog_c001', 'NVC0905_22_002_Ecog_c002',
                    'NVC0905_22_002_Ecog_c003', 'NVC0905_22_002_Ecog_c004',
                    'NVC0905_22_002_Ecog_c005', 'NVC0905_22_002_Ecog_c006',
                    'NVC0905_22_002_Ecog_c007', 'NVC0905_22_002_Ecog_c008',
                    'NVC0905_22_002_Ecog_c009', 'NVC0905_22_002_Ecog_c010',
                    'NVC0905_22_002_Ecog_c011', 'NVC0905_22_002_Ecog_c012',
                    'NVC0905_22_002_Ecog_c013', 'NVC0905_22_002_Ecog_c014',
                    'NVC0905_22_002_Ecog_c015', 'NVC0905_22_002_Ecog_c016'
                    ]

    # group channel names by number of channels (exclude remainder)
    num_channels = hparams['num_channels']

    # split channel names into sublists of length num_channels
    channel_groups = [all_ch_names[x:x + num_channels] for x in
                      range(0, len(all_ch_names) - len(all_ch_names) % num_channels, num_channels)]

    # select only one file
    selected_fname = 'Dog_1_ictal_segment_2.mat'

    # filter samples_df
    samples_df = samples_df.loc[:,
                 # samples_df['fname'] == selected_fname,
                 all_ch_names + ['time', 'fname']]

    # FILTER: keep only inter/ictal rows
    if hparams['selected_label_desc'] == 'interictal':
        samples_df = samples_df[samples_df['fname'].apply(lambda name: 'interictal' in name)]
    elif hparams['selected_label_desc'] == 'ictal':
        samples_df = samples_df[samples_df['fname'].apply(lambda name: 'interictal' not in name)]
    elif hparams['selected_label_desc'] == 'test':
        samples_df = samples_df[samples_df['fname'].apply(lambda name: 'test' in name)]
    else:
        raise ValueError()

    for fname, group in samples_df.groupby('fname'):
        for ch_names in channel_groups:
            print(f"beginning training with {fname}/{ch_names}")
            # task = Task.init(project_name=f"inference/pairs/{fname}", task_name=f"gp_{ch_names}", reuse_last_task_id=False)
            # task.connect(hparams)

            # set rng seed
            seed_everything(hparams['random_seed'])

            # setup training data
            train_x = Tensor(group['time'].values)
            train_y = Tensor(group[ch_names].values)
            train_dataloader = DataLoader(SingleSampleDataset(train_x, train_y), num_workers=0)

            # plot data sample
            plt.plot(train_x, train_y)
            plt.legend(labels=ch_names)
            plt.xlabel('time (s)')
            plt.title('Data Sample')
            plt.show()

            # initialize PyTorch Lightning model
            model = SingleSampleEEGGPModel(hparams, train_x, train_y)

            # plot prior samples
            for i in range(n_draws):
                plt.plot(train_x, model(train_x).sample())
            plt.xlabel('time (s)')
            plt.title('Prior')
            plt.show()

            # define trainer and fit model
            logger_dirpath = f"{config['PATH'][config['RESULTS_MACHINE']]['LIGHTNING_LOGS']}/{fname[:-4]}/{ch_names}"
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss",
                # dirpath=logger_dirpath,
                filename="gp_inference-{epoch:03d}-{train_loss:.2f}",
                save_top_k=1,
                mode="min",
                every_n_epochs=hparams['n_epochs']
            )
            # todo: add early stopping
            logger = CSVLogger(save_dir=logger_dirpath, name="gp_embedding", version=hparams['version'])
            trainer = Trainer(max_epochs=hparams['n_epochs'], log_every_n_steps=200, gpus=1, profiler=None,
                              callbacks=[checkpoint_callback], fast_dev_run=hparams['fast_dev_run'], logger=logger,
                              deterministic=True, enable_progress_bar=hparams['enable_progress_bar'])
            trainer.fit(model, train_dataloader)

            # plot posterior samples
            for i in range(n_draws):
                plt.plot(train_x, model(train_x).sample())
            plt.xlabel('time (s)')
            plt.title('Posterior')
            plt.show()

            if hparams['fast_dev_run']:
                sys.exit(0)
            # close task
            # task.close()
