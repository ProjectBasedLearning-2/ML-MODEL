import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.pipeline import Pipeline

# Load your dataset
df = pd.read_csv('D:/PBL/ml/ML-MODEL/Fertilizer Prediction (1).csv')

# Separate features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Define preprocessing steps
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, [3, 4])  # Apply OneHotEncoder to columns 3 and 4 (Soil Type and Crop Type)
    ],
    remainder='passthrough'  # Pass through the remaining columns as-is
)

# Define the classifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=0)

# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline as a pickle file
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
