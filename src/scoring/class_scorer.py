from typing import Dict

from sklearn.metrics import accuracy_score, f1_score


class ClassScorer:
    @classmethod
    def score(cls, y_true, y_pred) -> Dict[str, float]:
        return {
            'f1_score': f1_score(y_true, y_pred, average='macro'),
            'accuracy_score': accuracy_score(y_true, y_pred),
        }
