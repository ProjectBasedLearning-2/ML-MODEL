import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.pipeline import Pipeline

df = pd.read_csv('Fertilizer Prediction (1).csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#
# ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), ['Soil Type', 'Crop Type'])], remainder = 'passthrough')
# X = np.array(ct.fit_transform(X))
#
# le = LabelEncoder()
# y = le.fit_transform(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#
# sc = StandardScaler()
# X_train[:, 16:] = sc.fit_transform(X_train[:, 16:])
# X_test[:, 16:] = sc.transform(X_test[:, 16:])
#
# classifier = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, random_state = 0)
# classifier.fit(X_train, y_train)



# preprocessor = ColumnTransformer(
#     transformers=[
#         ('encoder', OneHotEncoder(), [3, 4]),
#         ('scaler', StandardScaler(), slice(16, None))  # Standardize numeric features
#     ],
#     remainder='passthrough'
# )
#
# # Define classifier
# classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=0)
#
# # Create pipeline
# pipeline = Pipeline([
#     ('preprocessor', preprocessor)
# ])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, [3,4])
    ])

classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=0)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pipeline.fit(X_train, y_train)

pickle.dump(pipeline , open("model.pkl", "wb"))
