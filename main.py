import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, RFE, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
from collections import Counter
from src.ML_functions import optimize_f1_threshold, plot_precision_recall_curve, evaluate_model

#ignore warnings
warnings.filterwarnings("ignore")

#load the dataset into memory
df=pd.read_csv("/Users/gabrielemia/Documents/My Project/data/preprocessing_df.csv", index_col=0).drop("customerID", axis=1)

#convert "Senior Citizen" column to appropriate data type
df["SeniorCitizen"]=df["SeniorCitizen"].astype(str)

#impute missing values in the "Total Charges" column
mean= np.mean(df["TotalCharges"])
df["TotalCharges"]=df["TotalCharges"].fillna(mean)

#split the dataset into training and test sets
X_train, X_test, y_train, y_test= train_test_split(df.drop("Churn", axis=1), df["Churn"], test_size=0.3, random_state=0)

#discretize numerical features and encode categorical variables
ct=ColumnTransformer([
    ("discretizer", KBinsDiscretizer(strategy="uniform"), [4, 17, 18]),
     ("encoding", OneHotEncoder(), [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16])
],
remainder="passthrough")
X_train= ct.fit_transform(X_train)
X_test=ct.transform(X_test)

#retrieve feature names after preprocessing transformations
ar1=ct.named_transformers_["discretizer"].get_feature_names_out()
ar2=ct.named_transformers_["encoding"].get_feature_names_out()
feature_names=np.concatenate((ar1, ar2), axis=0)
bins_range=ct.named_transformers_["discretizer"].bin_edges_

#encode the target variable
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)

#select the 10 features most correlated with the target
selector=SelectKBest(chi2, k=10)
X_train=selector.fit_transform(pd.DataFrame(X_train, columns=feature_names), y_train)
X_test=selector.transform(pd.DataFrame(X_test, columns=feature_names))
feature_names=selector.get_feature_names_out()

#apply SMOTE to address the severe class imbalance between Churn = 0 and Churn = 1
resampler=SMOTE(random_state=0)
X_train, y_train=resampler.fit_resample(pd.DataFrame(X_train, columns=feature_names), y_train)
feature_names=resampler.get_feature_names_out()

#define a pipeline with Recursive Feature Elimination to retain the top 6 informative features and specify a hyperparameter grid
lr=LogisticRegression(class_weight="balanced", solver="liblinear")
rfe=RFE(lr, n_features_to_select=6, step=2)
pipeline=Pipeline([
    ("rfe", rfe),
    ("lr", lr)
])
param_grid={
    "lr__penalty":["l1", "l2"],
    "lr__C":[x for x in np.arange(0.1, 1, 0.1)]
}

#optimize the model's hyperparameters
optimizer=GridSearchCV(pipeline,
param_grid=param_grid,
error_score="raise",
scoring="f1",
cv=10)

#use the "optimize_f1_threshold" function to train the model with the threshold that maximizes F1-score by balancing precision and recall
best_f1_lr=optimize_f1_threshold(optimizer, pd.DataFrame(X_train, columns=feature_names) , y_train, pd.DataFrame(X_test, columns=feature_names), y_test, 0.1, 1, 0.1)

#extract the actual names of features selected by the model
feature_names_lr=best_f1_lr[2].feature_names_in_

#compute SHAP values for each feature by processing the dataset in batches due to its size
data= shap.sample(X_train, 10)
explainer=shap.KernelExplainer(best_f1_lr[2].predict, data)
shap_values=explainer.shap_values(X_test)

#evaluate model
evaluate_model(best_f1_lr[2], pd.DataFrame(X_test, columns=feature_names_lr), y_test, target="classification")

#visualize the top 6 informative features selected by the model
shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names_lr), max_display=6)

plt.show()

