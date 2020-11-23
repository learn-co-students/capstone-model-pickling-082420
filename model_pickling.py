import pickle

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from src.models.custom_transformers import PrecipitationTransformer

# Load all data into X and y
antelope_df = pd.read_csv("antelope.csv")
X = antelope_df.drop("spring_fawn_count", axis=1)
y = antelope_df["spring_fawn_count"]

# Note: we are not doing a train-test split, since we already have a "final"
# model chosen based on some previous train-test split. We want the best possible
# model, so we fit with the entire training set.

# Instantiate a pipeline that performs all preprocessing steps
pipe = Pipeline(steps=[
    ("transform_precip", PrecipitationTransformer()),
    ("encode_winter", ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"),
         ["winter_severity_index"])
    ], remainder="passthrough"
    )),
    ("linreg_model", LinearRegression())
])

# Fit the pipeline on the full dataset
pipe.fit(X, y)

# Not needed, but print out the coefficients as a way to demonstrate that the
# model was successfully fitted
print("coefficients")
print(pipe.named_steps["linreg_model"].coef_)

# Save the fitted pipeline
with open("model.pkl", 'wb') as f:
    pickle.dump(pipe, f)