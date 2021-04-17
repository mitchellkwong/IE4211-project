# IE4211-project

## Final submission
The notebook `group11.ipynb` is used to generate the final submission file `group11.csv`. Note: The notebook uses cached predictions (explained below).

## Cached base model predictions
Our final model is an ensemble model which takes base model predictions as input. To reduce runtime for the grader's convenience, we have cached all predictions in the `predictions/` folder. This process can be inspected from the `Results.ipynb` notebook. Note: this notebook uses cached models (explained below)

## Cached base models
We ran 10-fold cross validated grid search over a large parameter space to tune each model. As this process takes a long time to run, each model is saved to the `models/` folder once tuned. The search procedure for each model can be inspected from their respective notebooks.
## Grid Search
Code for our grid search procedure can be found in `utils.py`. This file defines a thin wrapper around sklearn's cross validated grid search to fix common parameters and add some utility methods.
## Data Preprocessing
All data preprocessing steps can be found in the `data` module under `data/preprocessing.py`. The datasets defined in this file are used across our entire project.
