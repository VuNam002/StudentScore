from pathlib import Path

import pandas as pd
from models.LightGBM import train_lightgbm
from models.random_forest import train_random_forest
from models.ridge import train_ridge
from models.svm import train_svm
from models.xgboost_regressor import train_xgboost
from utils.artifacts import (
    build_run_info,
    create_run_dir,
    save_json,
    save_model,
)
from utils.evaluation import evaluate_model
from utils.preprocessing import split_data

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "StudentScore.xls"

TEST_SIZE = 0.2
RANDOM_STATE = 42


MODELS = {
    "svm": train_svm,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
    "ridge": train_ridge,
    "lightgbm": train_lightgbm,
}


def main():
    data = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = split_data(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    run_dir = create_run_dir(BASE_DIR)
    save_json(
        build_run_info(
            DATA_PATH,
            data,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            models=list(MODELS),
        ),
        run_dir / "run_info.json",
    )

    summary = []
    for name, train_fn in MODELS.items():
        print(f"Training {name}...")
        model = train_fn(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)

        save_model(model, run_dir / "model" / f"{name}.joblib")
        save_json(metrics, run_dir / f"{name}_metrics.json")

        summary.append({"model": name, **metrics})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
