from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

target = "writing score"
num_features = ["reading score", "math score"]
ord_features = ["parental level of education", "gender", "lunch", "test preparation course"]
nom_features = ["race/ethnicity"]

education_value = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
gender_value = ["male", "female"]
lunch_value = ["free/reduced", "standard"]
test_value = ["none", "completed"]

def split_data(data, target = target, test_size = 0.2, random_state = 42):
    x = data.drop(columns=[target])
    y = data[target]
    return train_test_split(x, y, test_size = test_size, random_state = random_state)

def build_preprocessing_pipeline():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[education_value, gender_value, lunch_value, test_value]))
    ])

    nom_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("ord", ord_pipeline, ord_features),
        ("nom", nom_pipeline, nom_features)
    ])

    return preprocessor


