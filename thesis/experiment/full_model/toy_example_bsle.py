import matplotlib.pyplot as plt
import numpy as np
import numpyro
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import seaborn as sns
import pandas as pd

from msc.plot_utils import set_size
from msc.config_utils import config

plt.style.use(['science', 'no-latex'])
plt.viridis()


# generate data
n_samples = 100 * np.array([600, 400, 1])
centers = np.array([[0], [3], [5]])
X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=1, random_state=0)

# discretize for histogram
bins = np.linspace(-8, 8, 100, endpoint=True)
count, bins = np.histogram(X, bins=bins)

# dataframize for histogram
# data_df = pd.DataFrame({"x": X.squeeze(), "class": y.squeeze()})
# sns.displot(data_df, x="x", hue="class", stat="probability")

# fit density estimation
gmm = GaussianMixture(2)
gmm.fit(X)

# plot density log likelihood
theta = bins
scores = gmm.score_samples(theta.reshape(-1, 1))


width = 478  # pt
fig, axes = plt.subplots(1, figsize=set_size(width))
axes.hist(X, bins=bins, label='data histogram', density=True, zorder=2)
axes.plot(theta, np.exp(scores), label='gmm likelihood scores', lw=2, zorder=2)
# plt.show()
ci = numpyro.diagnostics.hpdi(X, 0.95)
hdpi_x = np.linspace(ci[0], ci[1], 100).squeeze()
hdpi_scores = gmm.score_samples(hdpi_x.reshape(-1,1))

axes.fill_between(hdpi_x, np.exp(hdpi_scores), label="95% hpdi", color='r', alpha=0.5, zorder=3)
axes.legend()

plt.savefig(f"{config['path']['figures']}/method/hdpi.pdf", bbox_inches='tight')
plt.clf()
