# literate-octo-pancake
This template provides a general API interface to deploy machine learning code to production and development environments. The API exposes five functions to perform training, inference, and metrics calculations and save and load models.

The context for development is the problem in which text narratives must be assigned multiple descriptors or codes: muli-class, multi-label text classification.

**The following are example uses of each method in this package.**

1. Train model
```
def train(data: pd.DataFrame, data_side: pd.DataFrame, target: list) -> Model:
    """
    This function trains a model instance on labeled data and returns the trained model.
    
    Arguments:
        data: A pandas DataFrame containing feature information.
        data_side: A pandas DataFrame containing any side information that cannot be naturally joined to the data object. For 
            example, these might be descriptors of target labels.
        target: A list of labels, associated with each observation in the data object.
    
    Returns:
        model: A trained Model object.
    """
    model.fit(data, target)
    return model
    
trained_model = train(X_train, y_side_data, y_train)
```

2. Evaluate model performance
```
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
    y_pred = model.predict(data)
    metrics = classification_report(target, y_pred)
    return metrics

metrics_dict = metrics(X_test, y_test, trained_model)
```

3. Generate predictions
```
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

inferences_list = predict(X, trained_model)
```

4. Save model to directory
```
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

filename = time.strftime("%Y%m%d-%H%M%S")
save(trained_model, f'/models/{filename}')
```

5. Load model from directory
```
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
    
loaded_model = load(f'/models/{filename}')
```
