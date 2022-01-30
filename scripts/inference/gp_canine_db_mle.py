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

    # FILTER: keep only interictal rows
    samples_df = samples_df[samples_df['fname'].apply(lambda name: 'interictal' in name)]

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

            # plot prior samples
            for i in range(n_draws):
                plt.plot(train_x, model(train_x).sample())
            plt.xlabel('time (s)')
            plt.title('Prior')
            plt.show()

            # define trainer and fit model
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss",
                dirpath=f"{config['PATH']['LOCAL']['LIGHTNING_LOGS']}/{fname[:-4]}/{ch_name[-4:]}",
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
