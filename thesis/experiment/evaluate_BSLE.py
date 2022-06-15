import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.utils.estimator_checks import check_estimator

from msc import config
from msc.cache_handler import get_samples_df
from msc.estimators import BSLE



if __name__ == "__main__":
        
    # bsle = BSLE(thresh=0.4)
    # for estimator, check in check_estimator(bsle, generate_only=True):
    #     check(estimator)
        
    bsle = BSLE(thresh=0.05)
    # for estimator, check in check_estimator(bsle, generate_only=True):
    #     check(estimator)

    dataset_id = str(config['dataset_id'])
    t_max = config['t_max']

    # load samples_df
    samples_df = get_samples_df(dataset_id, with_events=True, with_time_to_event=True)
    samples_df['class'] = samples_df['time_to_event'].apply(lambda x: 1 if x < 5 else 0)

    # split train/test
    samples_df['set'] = samples_df['time'].apply(lambda t: 'train' if t < t_max else 'test')

    times = samples_df['time'].to_numpy()

    data = np.stack(samples_df['embedding'])  # type: ignore
    rng = np.random.default_rng(seed=42)
    
    train_X = np.stack(samples_df.loc[samples_df['set'] == 'train', "embedding"])  # type: ignore
    test_X = np.stack(samples_df.loc[samples_df['set'] == 'test', "embedding"])  # type: ignore
    test_y = samples_df.loc[samples_df['set'] == 'test', "class"].to_numpy()

    bsle.fit(train_X)
    prob_pred = bsle.predict_proba(test_X)

    x = 5
    print(brier_score_loss(test_y, prob_pred))



