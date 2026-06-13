import json
import matplotlib.pyplot as plt
import numpy as np

class DistributionVisualizer:
    def __init__(self, db_path):
        print(f"Loading database: {db_path}")
        with open(db_path, "r") as f:
            self.db_content = json.load(f)
            
        self.model_name = self.db_content.get("model_name", "Unknown Model")
        self.total_length = self.db_content["total_length"]
        self.surprises = self.db_content["surprises"]
        print(f"Loaded data for model: {self.model_name}")
        print(f"Total surprise tokens recorded: {len(self.surprises)}")

    def plot_single_surprise(self, token_pos, output_path="single_surprise.png"):
        """Plot the probability distribution for a specific surprise token position."""
        pos_str = str(token_pos)
        if pos_str not in self.surprises:
            print(f"Position {token_pos} is not a surprise token in the DB.")
            return
            
        data = self.surprises[pos_str]
        if isinstance(data, dict) and "distribution" in data and data["distribution"] is not None:
            distribution = np.array(data["distribution"])
            vocab_size = len(distribution)
            token_ids = np.arange(vocab_size)
            
            plt.figure(figsize=(12, 6))
            # Due to vocab size being very large, we can use a scatter or bar plot. Line plot is faster.
            plt.plot(token_ids, distribution, color="blue", alpha=0.7)
            plt.title(f"Probability Distribution at Surprise Position {token_pos}\nModel: {self.model_name}")
            plt.xlabel("Token ID")
            plt.ylabel("Probability")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path + f"/{(self.model_name).replace('/','_')}_single_surprise_pos_{token_pos}.png")
            print("Saved single surprise distribution to " + output_path + f"/{(self.model_name).replace('/','_')}_single_surprise_pos_{token_pos}.png")
        else:
            print(f"No distribution data available for position {token_pos}.")

    def plot_summed_surprise(self, output_path="summed_surprise.png"):
        """Plot the summed probability distribution over all surprise tokens."""
        summed_distribution = None
        count = 0
        for pos, data in self.surprises.items():
            if isinstance(data, dict) and "distribution" in data and data["distribution"] is not None:
                dist = np.array(data["distribution"])
                if summed_distribution is None:
                    summed_distribution = dist
                else:
                    summed_distribution += dist
                count += 1

        if summed_distribution is not None:
            vocab_size = len(summed_distribution)
            token_ids = np.arange(vocab_size)

            plt.figure(figsize=(12, 6))
            plt.plot(token_ids, summed_distribution, color="red", alpha=0.7)
            plt.title(f"Summed Probability Distribution over {count} Surprise Tokens\nModel: {self.model_name}")
            plt.xlabel("Token ID")
            plt.ylabel("Summed Probability")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path + f"/{(self.model_name).replace('/','_')}_summed_surprise.png")
            print("Saved summed surprise distribution to" + output_path + f"/{(self.model_name).replace('/','_')}_summed_surprise.png")
        else:
            print("No distribution data available in the DB.")

if __name__ == "__main__":
    db_file = "test_files/testmannual.db.json"
    
    try:
        visualizer = DistributionVisualizer(db_file)
        
        # Example 1: Plot the summed distribution
        visualizer.plot_summed_surprise("test_files")
        
        # Example 2: Find the first valid surprise token (after position 0) to plot
        first_valid_pos = None
        for pos, data in visualizer.surprises.items():
            if pos != "0" and isinstance(data, dict) and "distribution" in data and data["distribution"] is not None:
                first_valid_pos = pos
                break
                
        if first_valid_pos:
            visualizer.plot_single_surprise(first_valid_pos, f"test_files")
            
    except FileNotFoundError:
        print(f"Database file {db_file} not found. Please run the compressor script first.")
