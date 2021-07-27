import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def train(data: pd.DataFrame, data_side: pd.DataFrame, target: list, model: Model) -> Model:
    """
    This function trains a model instance on labeled data and returns the trained model.
    
    Arguments:
        data: A pandas DataFrame containing feature information.
        data_side: A pandas DataFrame containing any side information that cannot be naturally joined to the data object. For 
            example, these might be descriptors of target labels.
        target: A list of labels, associated with each observation in the data object.
        model: A Model instance or object ready for training.
    
    Returns:
        model: A trained Model object.
    """
    model.fit(data, target)
    return model

def predict(data: pd.DataFrame, model: Model) -> List:
    """
    This function accepts unlabeled data and model arguments to generate inferences. It returns a list of predictions.
    
    Arguments:
        data: A pandas DataFrame containing feature information.
        model: A Model object ready for inference.
    
    Returns:
        predictions: A list of predicted labels, one for each observation in the data input.
    """
    predictions = model.predict(data)
    return predictions

def metrics(data: pd.DataFrame, target:list, model: Model) -> Dict:
    """
    This function generates classification metrics for model predictions on labeled data. It returns the metrics as a dictionary.
    
    Arguments:
        data: A pandas DataFrame containing feature information.
        target: A list of labels, associated with each observation in the data object.
        model: A Model object ready for inference.
    
    Returns:
        metrics: A dictionary of model performance evaluation statistics, such as F1 score, specificity, etc.
    """
    y_pred = predict(data, model)
    metrics = classification_report(target, y_pred)
    return metrics

def save(model: Model, directory: str):
    """
    This function saves a model object in the specified directory.
    
    Arguments:
        model: A model object.
        directory: Desired model filepath, including filename and extension. Note: Beware of overwriting previously saved models.
    
    Returns:
        None.
    """
    pickle.dump(model, open(directory, 'wb'))
    return None 

def load(directory: str) -> Model:
    """
    This function loads a specific model from directory and returns the model.
    
    Arguments:
        directory: Model filepath, including filename and extension.
    
    Returns:
        model: A model object previously saved in directory.
    """
    model = pickle.load(open(directory, 'rb'))
    return model

