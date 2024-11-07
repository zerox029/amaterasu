def categorical_accuracy(predictions, labels):
    max_preds = predictions.argmax(dim = 1, keepdim = True)

    correct_count = 0
    for idx, prediction in enumerate(max_preds):
        if labels[idx][prediction.item()] == 1 and labels[idx][0] != 1:
            correct_count += 1

    non_labels = sum(label[0] != 1 for label in labels)

    return correct_count / non_labels