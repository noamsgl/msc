import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.utils.estimator_checks import check_estimator

from msc import config
from msc.cache_handler import get_samples_df
from msc.estimators import BSLE

# plt.style.use(['science', 'no-latex'])

SEC = 1
MIN = 60 * SEC
HOUR = 60 * MIN

def eval_unsupervised_bsle(train_X, test_X, test_y):
    bsle = BSLE(thresh=0.05)

    bsle.fit(train_X)
    prob_pred = bsle.predict_proba(test_X)
    print("eval_unsupervised_bsle")
    print(brier_score_loss(test_y, prob_pred))

def eval_weakly_supervised_bsle(train_X, train_events, test_X, test_times, test_y):
    bsle = BSLE(thresh=0.05)

    bsle.fit(train_X, y=None, prior_events=train_events)
    prob_pred = bsle.predict_proba(test_X, samples_times=test_times)
    print("eval_weakly_supervised_bsle")
    print(brier_score_loss(test_y, prob_pred))


if __name__ == "__main__":
    # bsle = BSLE()
    # for estimator, check in check_estimator(bsle, generate_only=True):
    #     check(estimator)

    dataset_id = str(config['dataset_id'])
    t_max = config['t_max']

    # load samples_df
    samples_df = get_samples_df(dataset_id, with_events=True, with_time_to_event=True)
    samples_df['class'] = samples_df['time_to_event'].apply(lambda x: 1 if x <= 30 * MIN else 0)
    samples_df['is_event'] = samples_df['time_to_event'].apply(lambda x: True if x==0 else False)
    print(f"class counts is {samples_df['class'].value_counts().to_numpy()}")
    # split train/test
    samples_df['set'] = samples_df['time'].apply(lambda t: 'train' if t < t_max else 'test')

    # get train_times
    train_events = samples_df.loc[(samples_df['is_event']) & (samples_df['set'] == 'train'), 'time']
    # train_events = samples_df.loc[(samples_df['is_event']), 'time']

    # get embeddings
    train_X = np.stack(samples_df.loc[samples_df['set'] == 'train', "embedding"])  # type: ignore
    test_X = np.stack(samples_df.loc[samples_df['set'] == 'test', "embedding"])  # type: ignore
    test_y = samples_df.loc[samples_df['set'] == 'test', "class"].to_numpy()
    test_times = samples_df.loc[samples_df['set'] == 'test', "time"].to_numpy()


    eval_unsupervised_bsle(train_X, test_X, test_y)
    eval_weakly_supervised_bsle(train_X, train_events, test_X, test_times, test_y)
