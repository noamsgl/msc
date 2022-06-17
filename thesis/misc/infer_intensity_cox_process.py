import numpy as np
import gpytorch

import math
import torch
import gpytorch
import pyro
import tqdm
import matplotlib.pyplot as plt

from msc import config
from msc.models.annotations import COXModel

SEC = 1
MIN = 60 * SEC
HOUR = 60 * MIN
DAY = 24 * HOUR

# load data, times, event times
embeddings = np.load(f"{config['path']['data']}/{config['dataset_id']}/embeddings.npy")
times = np.load(f"{config['path']['data']}/{config['dataset_id']}/times.npy")
event_times = np.load(f"{config['path']['data']}/{config['dataset_id']}/event_times.npy")

# normalize by days
times = times / DAY
event_times = event_times / DAY

# convert to tensors
embeddings = torch.tensor(embeddings).double()
times = torch.tensor(times).double()
event_times = torch.tensor(event_times).double()
max_time = torch.max(event_times)
quadrature_times = torch.linspace(0, max_time, 1024).double()

# initialize model
torch.set_default_dtype(torch.float64)
model = COXModel(event_times.numel(), max_time)


# initialize training
num_iter = 1024
num_particles = 32

def train(lr=0.01):
    optimizer = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True)
    infer = pyro.infer.SVI(model.model, model.guide, optimizer, loss=loss)

    model.train()
    loader = tqdm.tqdm(range(num_iter))
    for i in loader:
        loss = infer.step(event_times, quadrature_times)
        loader.set_postfix(loss=loss)

train()

wait = True

# Here's a quick helper function for getting smoothed percentile values from samples
def percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95]):
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]

    # Smooth the samples
    kernel = torch.full((1, 1, 5), fill_value=0.2)
    percentiles_samples = [
        torch.nn.functional.conv1d(percentile_sample.view(1, 1, -1), kernel, padding=2).view(-1)
        for percentile_sample in percentile_samples
    ]

    return percentile_samples

    # Get the average predicted intensity function, and the intensity confidence region
model.eval()
with torch.no_grad():
    function_dist = model(quadrature_times)
    intensity_samples = function_dist(torch.Size([1000])).exp() * model.mean_intensity
    lower, mean, upper = percentiles_from_samples(intensity_samples)

# Plot the predicted intensity function
fig, ax = plt.subplots(1, 1)
line, = ax.plot(quadrature_times, mean, label=r"pred $\lambda$")
ax.fill_between(quadrature_times, lower, upper, color=line.get_color(), alpha=0.5)
# ax.plot(quadrature_times, true_intensity_function(quadrature_times), "--", color="k", label=r"Pred. $\lambda$")
ax.legend(loc="best")
ax.set_xlabel("Time")
ax.set_ylabel("Intensity ($\lambda$)")
ax.scatter(event_times, torch.zeros_like(event_times), label=r"Observed Arrivals")
plt.savefig(f"{config['path']['figures']}/annotations/intensity.pdf", bbox_inches='tight')
