import matplotlib.pyplot as plt
import mlx.core as mx
from mlx_lm import load
import numpy as np

class EntropyVisualizer:
    def __init__(self, model_name="mlx-community/Qwen2.5-3B-4bit"):
        print(f"Loading MLX model for visualization: {model_name}")
        self.model, self.tokenizer = load(model_name)
        
    def visualize(self, input_text, output_image_path="entropy_visualization.png"):
        # 1. Tokenize input
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
        else:
            tokens = self.tokenizer(input_text, add_special_tokens=False)["input_ids"]
            
        if len(tokens) == 0:
            print("Empty input.")
            return
            
        input_ids = mx.array([tokens])
        
        # 2. Forward pass to get logits
        logits = self.model(input_ids) # Shape: (1, N, V)
        
        # Convert logits to probabilities using softmax
        # logits[0] shape is (N, V)
        probs = mx.softmax(logits[0], axis=-1)
        
        # Get predictions (argmax)
        predictions = mx.argmax(probs, axis=-1).tolist()
        
        # We'll collect data to plot
        token_indices = list(range(1, len(tokens)))
        pred_probs = []
        actual_probs = []
        is_surprise = []
        total_surprise_space = [] # 1 - P(predicted)
        
        for i in range(1, len(tokens)):
            actual_tok = tokens[i]
            pred_tok = predictions[i-1]
            
            # Probability of the predicted token
            p_pred = probs[i-1, pred_tok].item()
            pred_probs.append(p_pred)
            
            # Probability of the actual token
            p_actual = probs[i-1, actual_tok].item()
            actual_probs.append(p_actual)
            
            # Total surprise probability space: sum of probs of all tokens EXCEPT the predicted one
            # which is simply 1.0 - p_pred
            surprise_space = 1.0 - p_pred
            total_surprise_space.append(surprise_space)
            
            # Is it a surprise?
            if pred_tok != actual_tok:
                is_surprise.append(True)
            else:
                is_surprise.append(False)
                
        # 3. Plotting
        plt.figure(figsize=(15, 8))
        
        # Plot predicted probability
        plt.plot(token_indices, pred_probs, label="Predicted Token Probability", color="blue", alpha=0.7)
        
        # Plot total surprise space
        plt.plot(token_indices, total_surprise_space, label="Total Surprise Space (1 - P_pred)", color="orange", alpha=0.7)
        
        # Highlight surprise tokens
        surprise_x = [token_indices[idx] for idx, surprise in enumerate(is_surprise) if surprise]
        surprise_y = [pred_probs[idx] for idx, surprise in enumerate(is_surprise) if surprise]
        plt.scatter(surprise_x, surprise_y, color="red", label="Surprise Token Occurrence", zorder=5)
        
        plt.title("Token Probability and Entropy Visualization")
        plt.xlabel("Token Index")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_image_path)
        print(f"Visualization saved to {output_image_path}")

if __name__ == "__main__":
    visualizer = EntropyVisualizer("mlx-community/Qwen2.5-3B-4bit")
    
    # Read text from random_test_text.txt
    try:
        with open("random_test_text.txt", "r") as f:
            text = f.read()
            visualizer.visualize(text, "test_files/entropy_visualization.png")
    except FileNotFoundError:
        print("random_test_text.txt not found. Please provide a valid text file.")
