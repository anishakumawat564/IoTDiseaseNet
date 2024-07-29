from sklearn.linear_model import BayesianRidge
import numpy as np
from Evaluation import evaluation
from math import log
from scipy import linalg

def Model_BAYESIAN(X_train, y_train,X_test , y_test,sol):

    # Creating and training model
    model = BayesianRidge()
    model.fit(X_train, y_train)
    # Model making a prediction on test data
    prediction = model.predict(X_test)
    Eval = evaluation(prediction.reshape(-1,1), y_test)
    return np.asarray(Eval).ravel()