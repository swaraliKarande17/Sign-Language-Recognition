from transformers import MarianMTModel, MarianTokenizer	
import torch	
	
	
class GermanToEnglishTranslator:	
    """	
    Translates German gloss sequences to English text.	
    Uses Helsinki-NLP/opus-mt-de-en (MarianMT architecture).	
    Runs on CPU — no GPU required for inference.	
    Downloads model automatically on first use (~300MB).	
    """	
	
    def __init__(self, model_name="Helsinki-NLP/opus-mt-de-en"):	
        print("Loading translation model... (downloads on first run)")	
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)	
        self.model     = MarianMTModel.from_pretrained(model_name)	
        self.model.eval()	
        print("Translation model ready.")	
	
    def translate(self, gloss_sequence):	
        """	
        Translate a list of German gloss words to English.	
	
        Args:	
            gloss_sequence: list of strings	
                            e.g. ["HEUTE", "WETTER", "GUT"]	
	
        Returns:	
            English string e.g. "today the weather is good"	
        """	
        if not gloss_sequence:	
            return "(no signs detected)"	
	
        german_text = " ".join(gloss_sequence)	
	
        inputs = self.tokenizer(	
            [german_text],	
            return_tensors="pt",	
            padding=True,	
            truncation=True,	
            max_length=128	
        )	
	
        with torch.no_grad():	
            translated = self.model.generate(	
                **inputs,	
                num_beams=4,         # beam search for better quality	
                max_length=128,	
                early_stopping=True	
            )	
	
        result = self.tokenizer.decode(	
            translated[0], skip_special_tokens=True	
        )	
        return result	