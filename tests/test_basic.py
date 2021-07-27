import pytest
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report



def test_train(data: pd.DataFrame, data_side: pd.DataFrame, target: str):
    """
    This function Train on data with known labels, returns model
    
    Arguments:
    
    
    Returns:
    
    """
    model.fit(data, target)
    return model

def test_predict(data: pd.DataFrame, model: Model):
    """
    This function Generates predictions for unlabeled data, returns list of predictions
    
    Arguments:
    
    
    Returns:
    
    """
    predictions = model.predict(data)
    return predictions

def test_metrics(data: pd.DataFrame, target: str, model: Model):
    """
    This function Generates classification metrics, returns as dictionary
    
    Arguments:
    
    
    Returns:
    
    """
    y_pred = model.predict(data)
    metrics = classification_report(target, y_pred)
    return metrics

def test_save(model: Model, directory: str):
    """
    This function Saves model in specified directory (include filename and extension), returns None
    
    Arguments:
    
    
    Returns:
    
    """
    pickle.dump(model, open(directory, 'wb'))
    return None 

def test_load(directory: str):
    """
    This function Loads model from directory, returns model
    
    Arguments:
    
    
    Returns:
    
    """
    model = pickle.load(open(directory, 'rb'))
    return model
