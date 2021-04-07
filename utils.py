from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import GridSearchCV

class StandardizedGridSearchCV:
    """Utility class to standardize grid search procedure across all models. 
    
    Notes
    -----
    Setting can be overwritten by passing their new values as keyword arguments
    when initializing the class. Utiliy methods to print, save and load the
    underlying sklearn models are also defined."""
    
    DEFAULTS = dict( 
        scoring = [
            'neg_mean_squared_error',
            'r2',
        ],                                  # Metrics to track
        refit = 'neg_mean_squared_error',   # Used to select best candidate
        return_train_score = True,          # Keep train metrics to check overfitting
        cv = 10,                            # 10-fold cross validation
        n_jobs = -1,                        # Enable multiprocessing on all cores
        verbose = 10,                       # Print as much info as possible
    )
    
    def __init__(self, estimator, param_grid, **kwargs):
        self.model = GridSearchCV(
            estimator = estimator,
            param_grid = param_grid,
            **{**self.DEFAULTS, **kwargs}, # Overwrite defaults
        )

    def __repr__(self):
        """Pretty print to inspect model"""
        return grid_search_to_str(self.model)

    def __getattr__(self, name):
        """Inherit all other behaviour from grid search in self.model once it's fitted"""
        return getattr(self.model, name)

    @property
    def results(self):
        """Convert CV results to dataframe"""
        return extract_results(self.model)
    
    @classmethod
    def load(cls, filename):
        """Loads a saved GridSearchCV object from a pickle file"""
        path = Path(filename)
        with path.open('rb') as f:
            model = pickle.load(f)
        wrapper = cls(
            estimator = model.estimator,
            param_grid = model.param_grid,
        )
        wrapper.model = model
        return wrapper

    def save(self, filename):
        """Saves sklearn GridSearchCV object as pickle file"""
        path = Path(filename)
        root = Path(*path.parts[:-1])
        root.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self.model, f)

def grid_search_to_str(grid_search):
    """Pretty print an sklearn GridSearchCV object"""
    lines = [
        80 * '=',
        'Pipeline Structure',
        80 * '-',
        str(grid_search.estimator),
        '\n',
        80 * '=',
        'Parameter Space',
        80 * '-',
    ]

    for param, values in grid_search.param_grid.items():
        lines.append(param)
        for value in values:
            lines.append(f'- {value}')

    return '\n'.join(lines)

def extract_results(grid_search):
    """Extracts grid search results into a dataframe"""
    results = grid_search.cv_results_.copy()
    params = pd.DataFrame(results.pop('params'))
    values = pd.DataFrame(results)
    values = values.loc[:, ~values.columns.str.contains('param_')]
    df = pd.concat([params, values], axis=1)
    df = df.set_index(list(params.columns))
    df = df.sort_values('rank_test_neg_mean_squared_error')
    return df