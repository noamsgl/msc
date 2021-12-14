import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

fpath = r"/cs_storage/noamsi/results/epilepsiae/max_cross_corr/surfCO/pat_3500/20211213T182128/window_733.pkl"
feature_name = 'Maximal Cross Correlation'
patient_name = "pat_3500"
window_name = "w_733"
plt.title(f"{feature_name}\nfor {patient_name}, {window_name}")
X = pickle.load(open(fpath, 'rb'))

ax = plt.subplot()
im = ax.imshow(X)

plt.xlabel('time (5 s frames)')
plt.ylabel('index of channel pair')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()
