import json
import mlx.core as mx
from mlx_lm import load

class MLXCompressor:
    def __init__(self, model_name="mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
        print(f"Loading MLX model: {model_name}")
        self.model_name = model_name
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
        probs = mx.softmax(logits[0], axis=-1)
        
        # 3. Get predictions
        predictions_mx = mx.argmax(probs, axis=-1)
        mx.eval(probs, predictions_mx)
        predictions = predictions_mx.tolist()
        
        surprises = {}
        # Token 0 has no prior context, so it's always a surprise
        # We don't have a distribution for token 0, so we just store null for distribution
        surprises["0"] = {"token": tokens[0], "distribution": None}
        
        # For token i, the prediction comes from logits at i-1
        for i in range(1, len(tokens)):
            pred_tok = predictions[i-1]
            actual_tok = tokens[i]
            if pred_tok != actual_tok:
                # Save the full probability distribution over the vocabulary for this surprise position
                # Round to 6 decimals to save space in JSON
                distribution = [round(x, 2) for x in probs[i-1].tolist()]
                total_prob_norm_fact = sum(distribution)
                distribution = [round(x / total_prob_norm_fact, 2) for x in distribution]
                surprises[str(i)] = {"token": actual_tok, "distribution": distribution}
                
        db_content = {
            "model_name": getattr(self, "model_name", "unknown"),
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
        # Handle both old format (value is int) and new format (value is dict)
        surprises = {}
        for k, v in db_content["surprises"].items():
            if isinstance(v, dict):
                surprises[int(k)] = v["token"]
            else:
                surprises[int(k)] = v
        
        if total_length == 0:
            return ""
            
        current_token = surprises[0]
        output_tokens = [current_token]
        
        for i in range(1, total_length):
            # Forward pass over full sequence
            input_ids = mx.array([output_tokens])
            logits = self.model(input_ids)
            greedy_pred = mx.argmax(logits[0, -1, :], axis=-1).item()
            
            if i in surprises:
                next_token = surprises[i]
            else:
                next_token = greedy_pred
                
            output_tokens.append(next_token)
            
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(output_tokens)
        else:
            return ""
