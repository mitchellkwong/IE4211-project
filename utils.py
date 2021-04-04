import pandas as pd

def extract_results(grid_search):
    """Extracts grid search results into a dataframe, optionally moving scoring columns to the left"""
    results = grid_search.cv_results_.copy()
    params = pd.DataFrame(results.pop('params'))
    values = pd.DataFrame(results)
    values = values.loc[:, ~values.columns.str.contains('param_')]
    columns = [
        'mean_train_neg_mean_squared_error',
        'mean_test_neg_mean_squared_error',
    ]
    columns = columns + [col for col in values.columns if col not in columns]
    df = pd.concat([params, values[columns]], axis=1)
    df = df.set_index(list(params.columns))
    df = df.sort_values('rank_test_neg_mean_squared_error')
    return df