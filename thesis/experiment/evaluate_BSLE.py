import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator

from msc import config
from msc.cache_handler import get_samples_df
from msc.estimators import BSLE



if __name__ == "__main__":
        
    bsle = BSLE()
    check_estimator(bsle)

    dataset_id = str(config['dataset_id'])
    t_max = config['t_max']

    # load samples_df
    samples_df = get_samples_df(dataset_id)
    
    # split train/test
    samples_df['class'] = samples_df['time'].apply(lambda t: 'train' if t < t_max else 'test')

    times = samples_df['time'].to_numpy()

    data = np.stack(samples_df['embedding'])  # type: ignore
    rng = np.random.default_rng(seed=42)
    
    train_X = np.stack(samples_df.loc[samples_df['class'] == 'train', "embedding"])
    test_X = np.stack(samples_df.loc[samples_df['class'] == 'test', "embedding"])
    test_y = None

    bsle.fit(train_X)
    bsle.predict_proba(test_X)



