from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from utils.preprocessing import build_preprocessing_pipeline
from utils.tuning import grid_search


def train_lightgbm(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", LGBMRegressor(random_state=42, verbose=-1, n_jobs=1))

    ])
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1],
        "model__num_leaves": [7, 15, 31],
    }
    return grid_search(reg, param_grid, x_train, y_train)