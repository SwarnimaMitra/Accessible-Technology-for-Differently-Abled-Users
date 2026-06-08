from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class TextSimplifier:
    def __init__(self, model_name="t5-small"):
        self.device = torch.device("cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # The standard T5 needs to be prompted with "summarize: " or "simplify: "
        # For simplification we use a specific checkpoint or a prompt.
        # Here we use a simple approach: prepend "simplify: " (works after fine‑tuning)
        # For zero-shot, we use the "summarize" task as a fallback.

    def simplify(self, text, max_length=150):
        input_text = f"simplify: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
        simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not simplified:
            # fallback to summarization
            inputs = self.tokenizer(f"summarize: {text}", return_tensors="pt", truncation=True).to(self.device)
            outputs = self.model.generate(**inputs, max_length=max_length)
            simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simplified