from sklearn.pipeline import Pipeline
from utils.preprocessing import build_preprocessing_pipeline
from utils.tuning import grid_search
from xgboost import XGBRegressor


def train_xgboost(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1))
    ])
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1],
    }
    return grid_search(reg, param_grid, x_train, y_train)
