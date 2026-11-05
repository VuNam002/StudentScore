from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": mse ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }
