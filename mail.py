
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle as pkl


#CSV import
df=pd.read_csv("emails.csv")


# Separate features and target
X = df.drop(['Email No.', 'Prediction'], axis=1)
y = df['Prediction']

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model to a file

with open('rf_model.pkl', 'wb') as f:
    pkl.dump(rf_model, f)
print("done")
