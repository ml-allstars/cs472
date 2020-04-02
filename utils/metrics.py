def precision_recall(actual, predictions):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(actual)):
        if actual[i] == 1:
            if predictions[i] == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predictions[i] == 1:
                false_positives += 1
    precision = 0 if (true_positives + false_positives == 0) else true_positives / (true_positives + false_positives)
    recall = 0 if (true_positives + false_negatives == 0) else true_positives / (true_positives + false_negatives)
    return precision, recall
