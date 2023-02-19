# TODO: binary_classification_metrics in metrics.py
import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    
    precision = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
        
    recall = 0
    if tp + fn != 0:
        recall = tp / (tp + fn)
        
    f1 = 0
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
        
    accuracy = 0
    if tp + tn + fp + fn != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    
    return (y_true == y_pred).sum() / y_pred.shape[0]


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return 1 - ss_res / ss_tot


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return ((y_pred - y_true) ** 2).sum() / y_true.shape[0]


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return (y_pred - y_true).abs().sum() / y_true.shape[0]
    