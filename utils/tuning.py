from sklearn.model_selection import GridSearchCV
from tqdm_joblib import tqdm_joblib


def grid_search(pipeline, param_grid, x_train, y_train, cv=5, scoring="r2"):
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(x_train, y_train)

    n_candidates = 1
    for values in param_grid.values():
        n_candidates *= len(values)

    with tqdm_joblib(total=n_candidates * cv, desc="  GridSearch"):
        grid.fit(x_train, y_train)

    print("  Best params:", grid.best_params_)
    print(f"  Best CV {scoring}:", round(grid.best_score_, 4))
    return grid.best_estimator_
