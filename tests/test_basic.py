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


def test_train(data: pd.DataFrame, data_side: pd.DataFrame, target: str):
    """
    This function Train on data with known labels, returns model
    
    Arguments:
    
    
    Asserts:
    
    """
    model.fit(data, target)
    assert model

def test_predict(data: pd.DataFrame, model: Model):
    """
    This function Generates predictions for unlabeled data, returns list of predictions
    
    Arguments:
    
    
    Asserts:
    
    """
    predictions = model.predict(data)
    assert predictions

def test_metrics(data: pd.DataFrame, target: str, model: Model):
    """
    This function Generates classification metrics, returns as dictionary
    
    Arguments:
    
    
    Asserts:
    
    """
    y_pred = model.predict(data)
    metrics = classification_report(target, y_pred)
    assert metrics

def test_save(model: Model, directory: str):
    """
    This function Saves model in specified directory (include filename and extension), returns None
    
    Arguments:
    
    
    Asserts:
    
    """
    pickle.dump(model, open(directory, 'wb'))
    assert None 

def test_load(directory: str):
    """
    This function Loads model from directory, returns model
    
    Arguments:
    
    
    Asserts:
    
    """
    model = pickle.load(open(directory, 'rb'))
    assert model
