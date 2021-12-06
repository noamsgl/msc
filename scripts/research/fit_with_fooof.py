from fooof import FOOOF

from fooof.utils.download import load_fooof_data

freqs = load_fooof_data('freqs.npy', folder='data')

spectrum = load_fooof_data('spectrum.npy', folder='data')
