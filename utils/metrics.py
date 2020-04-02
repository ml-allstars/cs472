WEIGHTS = {
    'score': .33,
    'precision': .33,
    'recall': .33,
}


def precision_recall_score(actual, predictions):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_correct = 0

    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            total_correct += 1
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
    score = total_correct / len(actual)
    return precision, recall, score


def total_quality(results):
    precision, recall, score = results
    return WEIGHTS['precision'] * precision + WEIGHTS['recall'] * recall + WEIGHTS['score'] * score
