"""
Implimentation of logistic regression to classify the spambank dataset.
"""
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# get data
features = list(pd.read_csv("train_data.csv", nrows=1))
X = pd.read_csv("train_data.csv", usecols=[col for col in features if col != "class"])
y = pd.read_csv("train_data.csv", usecols=["class"])
test_data_df = pd.read_csv("test_data.csv")

# feature scaling
scalar = StandardScaler()
X = scalar.fit_transform(X)
test_data_df = scalar.fit_transform(test_data_df)

# get preds
clf = LogisticRegression().fit(X, y.values.ravel())
preds = clf.predict(test_data_df)
logging.info("Predicted labels: %s", preds)
