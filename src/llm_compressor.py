import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMCompressor:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        print(f"Loading model: {model_name}")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def compress(self, input_text, db_path):
        # 1. Tokenize input
        tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
        if len(tokens) == 0:
            print("Empty input.")
            return 0, 0
            
        input_ids = torch.tensor([tokens], device=self.device)
        
        # 2. Forward pass (Teacher Forcing - completely parallelized)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # Shape: (1, N, V)
            
        predictions = logits[0].argmax(dim=-1).cpu().numpy()
        
        surprises = {}
        # Token 0 has no prior context, so it's always a surprise
        surprises["0"] = tokens[0]
        
        # For token i, the prediction comes from logits at i-1
        for i in range(1, len(tokens)):
            pred_tok = predictions[i-1]
            actual_tok = tokens[i]
            if pred_tok != actual_tok:
                surprises[str(i)] = actual_tok
                
        db_content = {
            "total_length": len(tokens),
            "surprises": surprises
        }
        
        with open(db_path, "w") as f:
            json.dump(db_content, f)
            
        return len(tokens), len(surprises)

    def decompress(self, db_path):
        with open(db_path, "r") as f:
            db_content = json.load(f)
            
        total_length = db_content["total_length"]
        surprises = {int(k): v for k, v in db_content["surprises"].items()}
        
        if total_length == 0:
            return ""
            
        current_token = surprises[0]
        output_tokens = [current_token]
        
        input_ids = torch.tensor([[current_token]], device=self.device)
        past_key_values = None
        
        for i in range(1, total_length):
            # We must run a forward pass on the last input_id to update KV cache and get predictions
            with torch.no_grad():
                outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                greedy_pred = outputs.logits[0, -1, :].argmax(dim=-1).item()
                
            if i in surprises:
                next_token = surprises[i]
            else:
                next_token = greedy_pred
                
            input_ids = torch.tensor([[next_token]], device=self.device)
            output_tokens.append(next_token)
            
        return self.tokenizer.decode(output_tokens)
