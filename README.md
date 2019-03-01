The MFoM Framework for Speech Attribute Detection
=========================================

Python project for speech articulatory attributes detection, such as manner and place of articulation.

Project Structure
------------
```
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── envs               <- Environment settings files: Anaconda, Dockerfile
    ├── experiments
    │   ├── logs
    │   ├── params         <- Training settings, hyperparameters
    │   ├── submissions    <- Evaluation model results, submission to the challenge leaderboard
    │   ├── system         <- Trained and serialized models, model predictions, or model summaries
    │   └── experiment.py  <- Main file to run the particular experiment, it is based on the framework in 'src' folder
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Manuals, literature and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Main source code for use in this project. Framework structure.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    |
    ├── test_environment.py
    |
    ├── tests              <- Test framework code from 'src' folder
    │   └── data           <- data for testing
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

## Main Components

* TODO insert Architecture Image

### Pipeline

### Data Loader

### Trainer

### Model

## Working Behavior
* Connection between all components 

<p><small>Project is based on the <a target="_blank" href="https://github.com/Vanova/cookiecutter-data-science">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
