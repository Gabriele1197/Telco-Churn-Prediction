from sklearn.model_selection import FixedThresholdClassifier
from sklearn.metrics import f1_score, classification_report, PrecisionRecallDisplay, ConfusionMatrixDisplay, mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os

def optimize_f1_threshold(estimator, X_train, y_train, X_test, y_test, start, end, step=0.1, pos_label=1, response_method="predict_proba"):
    """
    Optimizes the F1-score by selecting the best classification threshold in the presence of class imbalance.

    Parameters
    ----------
    model : estimator object
        The model implementing either `predict_proba` or `decision_function`.

    X_train : array-like
        The training input data.

    y_train : array-like
        The training target values.

    X_test : array-like
        The test input data.

    y_test : array-like
        The test target values.

    start : float
        Starting value of the threshold range.

    end : float
        Final value of the threshold range.

    step : float, default=0.1
        Step size for threshold increments.

    pos_label : int or str, default=1
        The label of the positive class.

    response_method : {'auto', 'decision_function', 'predict_proba'}, default='auto'
        Determines how prediction scores are obtained from the model:
        - 'auto': Tries `predict_proba` then `decision_function` in order.
        - 'decision_function': Uses the model’s decision function.
        - 'predict_proba': Uses the probability estimates.

    Returns
    -------
    best_f1 : float
        The highest F1-score achieved.

    best_threshold : float
        The threshold that yields the best F1-score.

    best_model : sklearn.base.BaseEstimator
        The fitted model using the optimal threshold.
    """
    f1=[]
    for i in np.arange(start, end, step):
        model=FixedThresholdClassifier(estimator, threshold=i, pos_label=pos_label, response_method=response_method)
        fitted_model=model.fit(X_train, y_train)
        f1.append([f1_score(y_test, fitted_model.predict(X_test)), i, fitted_model])
    f1=max(f1)
    return f1
"-------------------------------"

def plot_precision_recall_curve(model, X, y, plot_chance_level=True):
    """
    Plots the Precision-Recall curve of a trained classifier.

    Parameters
    ----------
    model : estimator object
        A trained classifier supporting `predict_proba` or `decision_function`.

    X : array-like
        The input data.

    y : array-like
        The true target labels.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    display=PrecisionRecallDisplay.from_estimator(model, X, y, plot_chance_level=plot_chance_level)
    display.ax_.set_title("Precision Recall Curve")
"-------------------------------"

def evaluate_model(model, X, y, target):
    """
    Evaluates a trained model on a given dataset and returns performance metrics.

    Parameters
    ----------
    model : estimator object
        The trained model to evaluate.

    X : array-like
        The input data.

    y : array-like
        The ground truth target values.

    target : {'classification', 'regression'}
        Specifies the type of task to evaluate:
        - 'classification': prints the classification report and returns a ConfusionMatrixDisplay object.
        - 'regression': returns a dictionary with MSE, MAE, RMSE, and R² metrics.

    Returns
    -------
    ConfusionMatrixDisplay or dict
        - If `target='classification'`, returns a ConfusionMatrixDisplay object and prints the classification report.
        - If `target='regression'`, returns a dictionary containing the following metrics:
            - 'mse' : mean squared error
            - 'mae' : mean absolute error
            - 'rmse': root mean squared error
            - 'r2'  : R-squared (coefficient of determination)
    """
    assert target in ["classification", "regression"], "unknown target, please specify one of the following: classification, regression"
    if target=="classification":
        if os.getcwd()=="/Users/gabrielemia/Documents/My Project":
            print(classification_report(y, model.predict(X)))
            ConfusionMatrixDisplay.from_estimator(model, X, y).plot()
        else:
            print(classification_report(y, model.predict(X)))
            ConfusionMatrixDisplay.from_estimator(model, X, y)
    else:
        metrics={"R2":r2_score(y, model.predict(X)),
                 "Mean squared error":mean_squared_error(y, model.predict(X)),
                 "Root mean squared error":root_mean_squared_error(y, model.predict(X)),
                 "Mean absolute error":mean_absolute_error(y, model.predict(X))}
        print(metrics)