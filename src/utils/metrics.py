import editdistance	
import torch	
	
	
def decode_predictions(log_probs, idx2gloss):	
    """	
    Greedy CTC decode: at each time step pick the highest-probability token,	
    then collapse consecutive duplicates, then remove blank tokens (index 0).	
	
    Args:	
        log_probs: tensor (B, T, vocab_size)	
        idx2gloss: dict mapping integer index -> gloss string	
	
    Returns:	
        list of strings, one prediction per batch item	
        e.g. ["HEUTE WETTER GUT", "MORGEN REGEN"]	
    """	
    # Greedy: pick highest probability token at each timestep	
    preds = log_probs.argmax(dim=-1)  # (B, T)	
    results = []	
	
    for seq in preds:	
        decoded = []	
        prev = None	
        for token in seq.tolist():	
            # Skip duplicates and skip blank token (0)	
            if token != prev and token != 0:	
                decoded.append(idx2gloss.get(token, "<unk>"))	
            prev = token	
        results.append(" ".join(decoded))	
	
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