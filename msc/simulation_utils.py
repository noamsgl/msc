import torch
import matplotlib.pyplot as plt

class cycles_intensity_function(object):
  # model parameters
  intensity1, omega1, phi1 = 0.1, 1, 0  #daily
  intensity2, omega2, phi2 = 0.5, 1/7, 0  #weekly
  intensity3, omega3, phi3 = 1, 1/30, 0  #monthly

  def cycle1(self, times):
   return torch.cos(self.omega1 * times * 2 * math.pi + self.phi1).add(1).mul(0.5) * self.intensity1

  def cycle2(self, times):
    return torch.cos(self.omega2 * times * 2 * math.pi + self.phi2).add(1).mul(0.5) * self.intensity2

  def cycle3(self, times):
    return torch.cos(self.omega3 * times * 2 * math.pi + self.phi3).add(1).mul(0.5) * self.intensity3

  def __call__(self, times):
    return (self.cycle1(times) + self.cycle2(times) + self.cycle3(times))

  def plot_graph(self, times):
    plt.subplots(1,1, figsize=(8,4))
    plt.title("intensity function")
    plt.plot(times, self(times), label="intensity")
    plt.plot(times, self.cycle1(times), label="daily component")
    plt.plot(times, self.cycle2(times), label="weekly component")
    plt.plot(times, self.cycle3(times), label="monthly component")
    plt.xlabel("time")
    plt.legend()
