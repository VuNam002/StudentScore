from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from utils.preprocessing import build_preprocessing_pipeline


def train_random_forest(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    reg.fit(x_train, y_train)
    return reg
