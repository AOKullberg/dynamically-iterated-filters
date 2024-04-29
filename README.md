# Dynamically Iterated Filters: A unified framework for improved iterated filtering via smoothing
Code to reproduce the results of the title paper.

## Getting started
The easiest solution is to use `pipenv` for dependency management. 
A standard `requirements.txt` file is provided which can then be used in conjunction with `pipenv`.
The environment contains all necessary packages as well as Jupyter lab to run the notebooks.
##### Install Python 3.11
```
python3 -m pipenv --python 3.11
```
##### Install requirements
```
python3 -m pipenv install -r requirements.txt -e . --skip-lock
```
##### Run environment
```
python3 -m pipenv shell
```
##### Starting Jupyter lab
```
jupyter lab
```

## Reproducing the results
The notebooks should be self-explanatory for reproducing the results of the paper.
#### Tracking
Data of the MC runs are provided in `data/tracking`.
To re-run the MC trials, `run.py` with configuration is provided.
It can be run via Hydra multirun (in the venv) by
```
python run.py sim=trackingexample -m
```
This will produce a multirun directory with similar data to that in `data/tracking`.
The plots can then be reproduced using the `tracking-example-results.ipynb` notebook.
#### TDOA
The TDOA example is self-contained and can be reproduced following the `tdoa-example.ipynb` notebook.
#### 1D Example
The 1D example is contained in `damped_motivation.ipynb`.
#### Illustration of DIFs
The basic illustration of the inner workings of DIFs can be reproduced with `basic-dyniter-illustration.ipynb`. 
Note that this file contains a plethora of other things as well, but should be self-explanatory.