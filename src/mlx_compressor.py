import json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

class MLXCompressor:
    def __init__(self, model_name="mlx-community/Qwen2.5-3B-4bit"):
        print(f"Loading MLX model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        
    def compress(self, input_text, db_path):
        # 1. Tokenize input
        # Note: Depending on tokenizer, we may or may not want special tokens
        # Assuming similar behavior as the torch implementation
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
        else:
            # fallback for some tokenizers
            tokens = self.tokenizer(input_text, add_special_tokens=False)["input_ids"]
            
        if len(tokens) == 0:
            print("Empty input.")
            return 0, 0
            
        input_ids = mx.array([tokens])
        
        # 2. Forward pass (Teacher Forcing - completely parallelized)
        logits = self.model(input_ids) # Shape typically: (1, N, V)
        
        # 3. Get predictions
        predictions = mx.argmax(logits[0], axis=-1).tolist()
        
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
        
        input_ids = mx.array([[current_token]])
        
        # Initialize KV Cache
        cache = make_prompt_cache(self.model)
        
        for i in range(1, total_length):
            # Forward pass updates the cache
            logits = self.model(input_ids, cache=cache)
            greedy_pred = mx.argmax(logits[0, -1, :], axis=-1).item()
            
            if i in surprises:
                next_token = surprises[i]
            else:
                next_token = greedy_pred
                
            input_ids = mx.array([[next_token]])
            output_tokens.append(next_token)
            
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(output_tokens)
        else:
            return ""
