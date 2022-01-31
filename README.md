# Masters Research Repo

#### Author: Noam Siegel
#### Advisors: Oren Shriki, David Tolpin
#### School: Ben-Gurion University


## Index
The repository structure consists of:

* [scripts/](scripts/): main scripts
* [scripts/canine_db/](scripts/canine_db/): tools for manipulating Canine Epilepsy Dataset.
* [scripts/classification/](scripts/classification/): fitting classifiers to embeddings
* [scripts/density_estimation/](scripts/density_estimation/): fitting density estimators to embeddings
* [scripts/eda/](scripts/eda/): exploratory data analysis
* [scripts/generate/](scripts/generate/): generating synthetic EEG data
* [scripts/inference/](scripts/inference/): conditioning time-series models on data
* [scripts/psp/](scripts/psp/): scripts for the final project in physiological signal processing
* [scripts/reports/](scripts/reports/): scripts for dumping results to PDF reports
* [settings/](settings/): configuration file
* [msc/](msc/): a Python package with functions used across scripts


## Getting Started
### Dependencies

* python >= 3.8
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