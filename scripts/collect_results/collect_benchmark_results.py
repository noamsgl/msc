import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

benchmark_submissions_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\results\benchmarks\mhills\submission1645243931064-rf3000mss2Bfrs0_fft-with-time-freq-corr-1-48-r400-usf.csv"

benchmark_df = pd.read_csv(benchmark_submissions_fpath)

answer_key_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\data\seizure-detection\SzDetectionAnswerKey.csv"

answers_df = pd.read_csv(answer_key_fpath)

benchmark_df = pd.merge(answers_df, benchmark_df, on='clip', suffixes=('_truth', '_bench'))

RocCurveDisplay.from_predictions(benchmark_df['seizure_truth'], benchmark_df['seizure_bench'])
plt.show()

print(roc_auc_score(benchmark_df['seizure_truth'], benchmark_df['seizure_bench']))
