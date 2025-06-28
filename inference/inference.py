# inference/inference.py

import pickle
import numpy as np

def model_fn(model_dir):
    """Load model from the directory"""
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, content_type='text/csv'):
    """Parse input"""
    if content_type == 'text/csv':
        return np.array([float(x) for x in request_body.split(',')]).reshape(1, -1)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make prediction"""
    return model.predict(input_data)

def output_fn(prediction, content_type='text/csv'):
    """Format prediction output"""
    return str(prediction[0])
