import editdistance	
import torch	
	
	
def decode_predictions(log_probs, idx2word):
    """Greedy CTC decode — joins characters back to string."""
    results = []
    pred_ids = log_probs.argmax(-1)
    for seq in pred_ids:
        decoded, prev = [], None
        for p in seq.tolist():
            if p != 0 and p != prev:
                decoded.append(idx2word.get(p, ""))
            prev = p
        # Join chars directly (no spaces between them)
        results.append("".join(decoded))
    return results	
	
	
def word_error_rate(predictions, references):	
    """	
    Calculate Word Error Rate (WER) over a list of predictions.	
    Uses edit distance at the word level.	
	
    Args:	
        predictions: list of prediction strings	
        references:  list of ground-truth strings	
	
    Returns:	
        float: WER value (lower = better)	
    """	
    total_errors = 0	
    total_words  = 0	
	
    for pred, ref in zip(predictions, references):	
        pred_words = pred.strip().split()	
        ref_words  = ref.strip().split()	
        total_errors += editdistance.eval(pred_words, ref_words)	
        total_words  += len(ref_words)	
	
    return total_errors / max(total_words, 1)	