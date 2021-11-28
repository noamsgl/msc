# Masters Research Repo

#### Author: Noam Siegel
#### Advisors: Oren Shriki, David Tolpin
#### School: Ben-Gurion University


## Index

This repository consist of:

* [msc.bocd](msc/bocd): an implementation of Bayesian Online Changepoint Detection.
* [msc.data_utils](msc/data_utils): utilities to load EEG datasets.
* [msc.forecaster](msc/forecaster): utilities to train a probabilistic forecasting model. 
* [notebooks](notebooks): Jupyter notebooks with examples.
* [scripts](scripts): Jupyter notebooks with examples.

## Examples
Here are some demonstrations of what you can do with msc.

* [Load EEG dataset](notebooks/demos/load_eeg_data.ipynb)
* [Run GP inference on cpu](notebooks/demos/demo_GP_workflow.ipynb)
* [Run GP inference on gpu](notebooks/demos/demo_GP_workflow_gpu.ipynb)
* [Run BOCD algorithm](notebooks/demos/simple_bayesian_online_changepoint_detection.ipynb)

## Getting Started
### Dependencies

* python >= 3.6
* pipenv

### Installing

* `$ pipenv shell` to activate virtual environment
* `$ pipenv install` to install project [dependencies](Pipfile)


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

* My advisors - Oren and David.
* Ben Gurion University.
* Project dependencies.
* [Eyal](http://www.epilepsy.org.il/) - Israeli Epilepsy Association 
