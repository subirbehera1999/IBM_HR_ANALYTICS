# Import required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import classification_report,f1_score,accuracy_score
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import pickle
from threshold.tuning import find_best_threshold_f1
from threshold.threshold_classifier import ThresholdClassifier

# Import dataset
df = pd.read_csv("data/attrition_data.csv")
df["Attrition"] = df["Attrition"].map({"No":0,"Yes":1})
df = df.drop(['EmployeeCount','Over18','StandardHours','EmployeeNumber'],axis=1)

# Separate X and y variable
x = df.drop("Attrition",axis=1)
y = df["Attrition"]

# Column groups
category_cols=x.select_dtypes(include="object").columns.to_list()
numeric_cols = x.select_dtypes(include="int").columns.to_list()

# Preprocessing with column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot",OneHotEncoder(drop="first",handle_unknown="ignore"),category_cols),
        ("robust_scale",RobustScaler(),numeric_cols)
    ]
)

# Define model with pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000,class_weight='balanced',solver='liblinear'))
])


# Tran test split
X_train, X_val_test, y_train, y_val_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test, random_state=42)


# fit model
model.fit(X_train,y_train)
calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(model),method="isotonic")

calibrated_model.fit(X_val, y_val)

y_val_prob = calibrated_model.predict_proba(X_val)[:,1]
best_threshold = find_best_threshold_f1(y_val,y_val_prob)

final_model = ThresholdClassifier(model=calibrated_model,threshold=best_threshold)
final_model.fit(X_train,y_train)


# save the model 
with open("models/model.pkl", "wb") as f:
    pickle.dump(final_model, f)