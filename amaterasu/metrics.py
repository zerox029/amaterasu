from typing import Any
from sklearn.metrics import confusion_matrix
from itertools import chain

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def remove_padding_and_argmax(predictions, labels) -> tuple[list[Any], list[Any]]:
    clean_preds = []
    clean_labels = []
    for i, label in enumerate(labels):
        if label[0] != 1:
            clean_labels.append(label.argmax(dim=0, keepdim=True).item())
            clean_preds.append(predictions[i].argmax(dim=0, keepdim=True).item())

    return clean_labels, clean_preds


def true_positive_count(predictions, labels, class_id: int = -1) -> int:
    """Get the number of true positive predictions from a list of predicted and actual classes"""
    true_positives = 0

    for idx, prediction in enumerate(predictions):
        if class_id != -1:
            if prediction != class_id or labels[idx] != class_id:
                continue

        if prediction == labels[idx]:
            true_positives += 1

    return true_positives

def false_positive_count(predictions, labels, class_id: int) -> int:
    """Get the number of samples predicted as class_id that were actually of a different class"""
    false_positives = 0

    for idx, prediction in enumerate(predictions):
        if prediction != class_id:
            continue

        if prediction != labels[idx]:
            false_positives += 1

    return false_positives

def false_negative_count(predictions, labels, class_id: int) -> int:
    """Get the number of samples not predicted as class_id that should have been predicted as class_id"""
    false_negatives = 0

    for idx, prediction in enumerate(predictions):
        if prediction == class_id:
            continue

        if labels[idx] == class_id:
            false_negatives += 1

    return false_negatives

def compute_precision(predictions, labels, class_id: int) -> float:
    """TP/(TP+FP)"""
    true_positives = true_positive_count(predictions, labels, class_id)
    false_positives = false_positive_count(predictions, labels, class_id)

    return true_positives/(true_positives + false_positives)

def compute_recall(predictions, labels, class_id: int) -> float:
    """TP/(TP+FN)"""
    true_positives = true_positive_count(predictions, labels, class_id)
    false_negatives = false_negative_count(predictions, labels, class_id)

    return true_positives/(true_positives + false_negatives)

def categorical_accuracy(predictions, labels) -> float:
    true_positives = true_positive_count(predictions, labels)

    return true_positives / len(predictions)

def f1(predictions, labels) -> tuple[float, list[float]]:
    """Computes the F1 score for each category as well as the average F1 score across all classes"""
    average_f1_score = 0
    categorical_f1_score = []
    for i in range(1, 5):
        precision = compute_precision(predictions, labels, i)
        recall = compute_recall(predictions, labels, i)

        f1_score = (2*precision*recall)/(precision+recall)
        average_f1_score += f1_score / 4
        categorical_f1_score.append(f1_score)

    return average_f1_score, categorical_f1_score

def generate_confusion_matrix(predictions, labels):
    for i in range(len(predictions)):
        labels[i], predictions[i] = remove_padding_and_argmax(predictions[i], labels[i])

    cm = confusion_matrix(list(chain.from_iterable(labels)), list(chain.from_iterable(predictions)))

    categories = ['PAD', 'S', 'B', 'E', 'I']

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(ticks=np.arange(len(categories)) + 0.5, labels=categories)
    plt.yticks(ticks=np.arange(len(categories)) + 0.5, labels=categories)
    plt.show()

def compute_metrics(predictions, labels) -> tuple[float, tuple[float, list[float]]]:
    """Computes the categorical accuracy and the f1 score for a set of predictions and labels"""
    predictions, labels = remove_padding_and_argmax(predictions, labels)
    return categorical_accuracy(predictions, labels), f1(predictions, labels)