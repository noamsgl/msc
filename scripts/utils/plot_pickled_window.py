import pickle

import matplotlib.pyplot as plt

fpath = r"/cs_storage/noamsi/results/epilepsiae/max_cross_corr/surfCO/pat_3500/20211213T182128/window_126.pkl"
X = pickle.load(open(fpath, 'rb'))

plt.imshow(X)
plt.show()