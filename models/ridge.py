from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from utils.preprocessing import build_preprocessing_pipeline
from utils.tuning import grid_search


def train_ridge(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", Ridge())
    ])
    param_grid = {
        "model__alpha": [0.01, 0.1, 1, 10, 100],
    }
    return grid_search(reg, param_grid, x_train, y_train)
