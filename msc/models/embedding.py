

import os
import hydra
from pytorch_lightning import seed_everything


class GPEmbeddor:
    def __init__(self, random_seed, num_channels, learning_rate, n_epochs, fast_dev_run, resume_from_checkpoint=False) -> None:
        self.random_seed = random_seed
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.fast_dev_run = fast_dev_run
        
        # Set seed for random number generators in pytorch, numpy and python.random
        seed_everything(random_seed, workers=True)

        # convert relative ckpt path to absolute path if necessary
        ckpt_path = resume_from_checkpoint
        if resume_from_checkpoint and not os.path.isabs(ckpt_path):
            resume_from_checkpoint = os.path.join(
                hydra.utils.get_original_cwd(), ckpt_path
            )

        # Init lightning datamodule
        # log.info
        # TODO: stopped here 


    def embed(self, data):
        # init GP model
        # transform data to low dimensional vector
        # save low dimensional vector to file
        raise NotImplementedError()


