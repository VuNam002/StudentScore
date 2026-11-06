from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from utils.preprocessing import build_preprocessing_pipeline
from utils.tuning import grid_search


def train_svm(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", SVR())
    ])
    param_grid = {
        "model__C": [1, 10, 100],
        "model__epsilon": [0.1, 0.5, 1.0],
        "model__kernel": ["rbf", "linear"],
    }
    return grid_search(reg, param_grid, x_train, y_train)
