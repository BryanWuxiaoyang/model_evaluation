from difflib import SequenceMatcher


class RecognitionMetric:
    """
    Evaluating recognition models by edit-distance ratio
    """
    def __init__(self):
        self.score = 0.0
        self.total_score = 0.0

    def append(self, pattern: str, text: str):
        ratio = SequenceMatcher(None, pattern, text).ratio()
        self.score += ratio
        self.total_score += 1.0

    def compute(self) -> float:
        return self.score / self.total_score if self.total_score != 0.0 else 0.0
