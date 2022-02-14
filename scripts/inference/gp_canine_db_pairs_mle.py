import matplotlib.pyplot as plt
from clearml import Task
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader

from msc import config
from msc.dataset import DogDataset, SingleSampleDataset
from msc.models import SingleSampleEEGGPModel

if __name__ == '__main__':
    # override parameters with provided dictionary
    hparams = {'random_seed': 42,
               'learning_rate': 1e-2,
               'n_epochs': 1000}

    n_draws = 8  # number of samples to draw for plotting
    dataset_dir = r"C:\Users\noam\Repositories\noamsgl\msc\data\seizure-detection\Dog_1"
    dataset = DogDataset(dataset_dir)
    samples_df = dataset.samples_df

    # select only one file
    selected_fname = 'Dog_1_ictal_segment_2.mat'

    # select a subset of channels
    selected_ch_names = ['NVC0905_22_002_Ecog_c001', 'NVC0905_22_002_Ecog_c002',
                         'NVC0905_22_002_Ecog_c003', 'NVC0905_22_002_Ecog_c004',
                         'NVC0905_22_002_Ecog_c005', 'NVC0905_22_002_Ecog_c006',
                         'NVC0905_22_002_Ecog_c007', 'NVC0905_22_002_Ecog_c008',
                         'NVC0905_22_002_Ecog_c009', 'NVC0905_22_002_Ecog_c010',
                         'NVC0905_22_002_Ecog_c011', 'NVC0905_22_002_Ecog_c012',
                         'NVC0905_22_002_Ecog_c013', 'NVC0905_22_002_Ecog_c014',
                         'NVC0905_22_002_Ecog_c015', 'NVC0905_22_002_Ecog_c016'
                         ]

    # group channel names into pairs
    paired_ch_names = list(zip(selected_ch_names[::2], selected_ch_names[1::2]))

    # filter samples_df
    samples_df = samples_df.loc[:,
                 # samples_df['fname'] == selected_fname,
                 selected_ch_names + ['time', 'fname']]

    # FILTER: keep only inter/ictal rows
    samples_df = samples_df[samples_df['fname'].apply(lambda name: 'interictal' in name)]

    # FILTER: skip completed tasks
    completed_fnames = [
        'Dog_1_interictal_segment_104.mat',
        'Dog_1_interictal_segment_103.mat',
        'Dog_1_interictal_segment_102.mat',
        'Dog_1_interictal_segment_101.mat',
        'Dog_1_interictal_segment_100.mat',
        'Dog_1_interictal_segment_10.mat',
        'Dog_1_interictal_segment_1.mat',
    ]
    samples_df = samples_df[samples_df['fname'].apply(lambda name: name not in completed_fnames)]

    for fname, group in samples_df.groupby('fname'):
        for pair_ch_names in paired_ch_names:
            pair_ch_names = list(pair_ch_names)
            print(f"beginning training with {fname}/{pair_ch_names}")
            task = Task.init(project_name=f"inference/pairs/{fname}", task_name=f"gp_{pair_ch_names}")
            task.connect(hparams)

            # set rng seed
            seed_everything(hparams['random_seed'])

            # setup training data
            train_x = Tensor(group['time'].values)
            train_y = Tensor(group[pair_ch_names].values)
            train_dataloader = DataLoader(SingleSampleDataset(train_x, train_y), num_workers=0)

            # plot data sample
            plt.plot(train_x, train_y)
            plt.legend(labels=pair_ch_names)
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
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss",
                dirpath=f"{config['PATH']['LOCAL']['LIGHTNING_LOGS']}/{fname[:-4]}/{pair_ch_names[-4:]}",
                filename="gp_inference-{epoch:03d}-{train_loss:.2f}",
                save_top_k=1,
                mode="min",
                every_n_epochs=1000
            )

            trainer = Trainer(max_epochs=hparams['n_epochs'], log_every_n_steps=1, gpus=1, profiler=None,
                              callbacks=[checkpoint_callback], fast_dev_run=False)
            trainer.fit(model, train_dataloader)

            # plot posterior samples
            for i in range(n_draws):
                plt.plot(train_x, model(train_x).sample())
            plt.xlabel('time (s)')
            plt.title('Posterior')
            plt.show()

            # close task
            task.close()
