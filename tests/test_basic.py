# Multiple test functions may apply to a given method, or vice versa.
## Do the methods work on ideal data, malformed data, and trivial and complicated models?
## Do not bundle too many assert statements, as this may complicate debugging.
## Ideal code coverage for tests is ~80-90%.
# Pytest looks for functions starting with ```test_```.
# The command to run pytest is simply ```pytest```, or ```make test``` if including a Makefile in the package.

import pytest
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def test_malformed_data(data: pd.DataFrame, data_side: pd.DataFrame, target: list):
    """
    This function tests the format of input data.
    
    Arguments:
        data: Features dataset as pandas DataFrame.
        target: List of corresponding labels.
    
    Asserts:
        Number of labels matches number of records.
    """
    assert len(data) == len(target)

def test_train_trivial_model(data: pd.DataFrame, target: list, model: Model):
    """
    This function tests the train function on a trivial model.
    
    Arguments:
        data: Features dataset as pandas DataFrame.
        target: List of corresponding labels.
        model: A model instance or object ready to be trained.
    
    Asserts:
        Trained model object is not None.
        Trained model coefficients are not all zero.
    """
    X_train = pd.Dataframe({"feature_1": np.random.random(5), "feature_2": np.random.random(5)})
    y_train = ['a','b','c','b','a']
    trained_model = train(X_train, y_train, model)
    assert trained_model is not None
    assert trained_model.coef_ != np.array([0,0,0,0,0])

def test_score_pred_trivial_model(data: pd.DataFrame, target: list, model: Model):
    """
    This function tests the metrics and predict functions on a trivial model.
    
    Arguments:
        data: Features dataset as pandas DataFrame.
        target: List of corresponding labels.
        model: A trained model object.
    
    Asserts:
        The number of inferences equals the number of required labels.
        The number of inference scores equals the number of inferences.
    """
    X_test = pd.Dataframe({"feature_1": np.random.random(5), "feature_2": np.random.random(5)})
    y_test = ['c','b','c','b','a']     
    metrics_dict = metrics(X_test, y_test, trained_model)
    inferences_list = predict(X_test, trained_model)
    assert len(inferences_list) == len(y_test)
    assert len(metrics_dict) == len(inferences_list)

# This function still needs to be worked out
def test_complicated_model(data: pd.DataFrame, data_side: pd.DataFrame, target: list, model: Model):
    """
    This function tests the train, predict, and metrics functions on a complicated model.
    
    Arguments:
        data:
        data_side:
        target:
        model: 
    
    Asserts:
        
    """
    predictions = model.predict(data)
    assert predictions

def test_save_load(model: Model, directory: str):
    """
    This function tests the save and load functions for a simple model object.
    
    Arguments:
        model: A model object.
        directory: Desired model filepath, including filename.
    
    Asserts:
        Model that has been saved can be retrieved as an object.
    """
    # This is a Mock model object from sklearn.
    model = RandomForestClassifier()
    filename = time.strftime("%Y%m%d-%H%M%S")
    save(model, f'/models/{filename}')
    retrieved_model = load(f'/models/{filename}')
    assert retrieved_model is not None


