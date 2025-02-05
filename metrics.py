import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def calculate_bleu(references, hypothesis):
    """
    Calculate BLEU-1,2,3,4 scores
    """
    if not isinstance(references, list):
        references = [references]
    
    # Tokenize reference and hypothesis
    tokenized_refs = [[list(ref)] for ref in references]
    tokenized_hyp = list(hypothesis)
    
    # Calculate BLEU scores for different n-grams
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for i in range(1, 5):
        try:
            score = sentence_bleu(
                tokenized_refs[0], 
                tokenized_hyp,
                weights=tuple([1.0/i]*i),
                smoothing_function=smoothing
            )
        except:
            score = 0.0
        bleu_scores.append(score)
    
    return bleu_scores

class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.bleu_scores = [[] for _ in range(4)]
    
    def update(self, references, hypotheses):
        for ref, hyp in zip(references, hypotheses):
            scores = calculate_bleu(ref, hyp)
            for i, score in enumerate(scores):
                self.bleu_scores[i].append(score)
    
    def get_metrics(self):
        metrics = {}
        for i in range(4):
            metrics[f'bleu-{i+1}'] = np.mean(self.bleu_scores[i])
        return metrics
