from typing import Dict, Any

def calculate_precision(true_positives: int, false_positives: int) -> float:
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)

def calculate_recall(true_positives: int, false_negatives: int) -> float:
    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)

def calculate_f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int
) -> Dict[str, Any]:
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    f1_score = calculate_f1_score(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }