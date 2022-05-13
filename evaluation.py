
def get_eval_metrics(predicted_classes, true_class_labels):
    """
    Returns evaluation metrics for binary classification problem.
    (1 is the positive class, 0 is the negative class)

    :predicted_classes: a list of binary class predictions (1 or 0)
    :true_class_labels: a list of true classes (1 or 0)

    :returns: accuracy, precision, recall
    """
    "*** Your Code Here ***"
    # these are counters for each occurrences
    T_neg = 0
    F_neg = 0
    T_pos = 0
    F_pos = 0
    total = len(predicted_classes)

    for i in range(total):
        # count which each case belongs to which class
        if predicted_classes[i] == 0 and true_class_labels[i] == 1:
            F_neg += 1
        elif predicted_classes[i] == 0 and true_class_labels[i] == 0:
            T_neg += 1
        elif predicted_classes[i] == 1 and true_class_labels[i] == 0:
            F_pos += 1
        else:
            T_pos += 1

    "*** Given part of code ***"
    return float((T_pos + T_neg)) / float(total), float(T_pos) / float((T_pos + F_pos)), float(T_pos) / float((T_pos + F_neg))
