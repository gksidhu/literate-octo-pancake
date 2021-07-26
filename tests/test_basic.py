from .context import sample
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import pickle
import pytest


def check_train(data: pandas.DataFrame, data_side: pandas.DataFrame, target: str):
    """Train on data with known labels, returns model"""
    model.fit(data, target)
    return model

def check_predict(data: pandas.DataFrame, model: Model):
    """Generates predictions for unlabeled data, returns list of predictions"""
    predictions = model.predict(data)
    return predictions

def check_metrics(data: pandas.DataFrame, target: str, model: Model):
    """Generates classification metrics, returns as dictionary"""
    y_pred = model.predict(data)
    metrics = classification_report(target, y_pred)
    return metrics

def check_save(model: Model, directory: str):
    """Saves model in specified directory (include filename and extension), returns None"""
    pickle.dump(model, open(directory, 'wb'))
    return None 

def check_load(directory: str):
    """Loads model from directory, returns model"""
    model = pickle.load(open(directory, 'rb'))
    return model
