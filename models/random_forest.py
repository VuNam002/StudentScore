from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from utils.preprocessing import build_preprocessing_pipeline
from utils.tuning import grid_search


def train_random_forest(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", RandomForestRegressor(random_state=42))
    ])
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }
    return grid_search(reg, param_grid, x_train, y_train)
