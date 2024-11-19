def categorical_accuracy(predictions, labels) -> float:
    true_positives = true_positive_count(predictions, labels)

    non_labels = sum(label[0] != 1 for label in labels)

    return true_positives / non_labels

def f1(predictions, labels) -> float:
    true_positives = true_positive_count(predictions, labels)
    return 0.0

def true_positive_count(predictions, labels) -> int:
    """Get the number of true positive predictions from a list of predicted and actual classes"""
    max_preds = predictions.argmax(dim=1, keepdim=True)

    true_positives = 0
    for idx, prediction in enumerate(max_preds):
        if labels[idx][prediction.item()] == 1 and labels[idx][0] != 1:
            true_positives += 1

    return true_positives

def recall(predictions, labels, class_id: int) -> int:
    """Get the recall value for a given class from a list of predicted and actual classes. """
    max_preds = predictions.argmax(dim=1, keepdim=True)

    correct_predictions = 0
    for idx, prediction in enumerate(max_preds):
        if labels[idx][prediction.item()] == 1 and labels[idx][0] != 1 and prediction.item() == class_id:
            correct_predictions += 1
