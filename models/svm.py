from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from utils.preprocessing import build_preprocessing_pipeline


def train_svm(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", SVR(kernel="rbf", C=1.0))
    ])
    reg.fit(x_train, y_train)
    return reg
