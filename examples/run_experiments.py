import os
import sys
import torch
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from instruct_prompt_optimizer import InstructPromptOptimizer

def run_experiments():
    optimizer = InstructPromptOptimizer(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    
    tasks = [
        {
            "name": "Parsing / Extraction (Generalization Test)",
            "dataset": [
                {
                    "user_prompt": "Extract the user's age and city from this text: 'Hi I am John, I'm 32 and I live in Seattle.'",
                    "target": "{\"age\": 32, \"city\": \"Seattle\"}"
                },
                {
                    "user_prompt": "Extract the user's age and city from this text: 'My name is Sarah. I reside in New York and I just turned 28.'",
                    "target": "{\"age\": 28, \"city\": \"New York\"}"
                },
                {
                    "user_prompt": "Extract the user's age and city from this text: 'Greetings, I am 45 years old and my home is in Chicago.'",
                    "target": "{\"age\": 45, \"city\": \"Chicago\"}"
                }
            ],
            "holdout_test": {
                "user_prompt": "Extract the user's age and city from this text: 'I am Alice, based in London, and I am 30 years old.'",
                "target": "{\"age\": 30, \"city\": \"London\"}"
            },
            "init_prompt": "json extract age city please now fast do it",
            "num_iters": 500,
            "batch_size": 32,
            "topk": 128,
            "patience": 50
        },
        {
            "name": "Math / Logic (Step-by-step)",
            "dataset": [
                {
                    "user_prompt": "What is 15 + 27?",
                    "target": "To find the sum of 15 and 27, we add 5 and 7 to get 12, carry the 1, and add 1, 1, and 2 to get 4. The result is 42."
                },
                {
                    "user_prompt": "What is 8 * 9?",
                    "target": "To find the product of 8 and 9, we multiply the two numbers. The result is 72."
                },
                {
                    "user_prompt": "What is 100 - 45?",
                    "target": "To find the difference between 100 and 45, we subtract 45 from 100. The result is 55."
                }
            ],
            "holdout_test": {
                "user_prompt": "What is 12 * 12?",
                "target": "To find the product of 12 and 12, we multiply the two numbers. The result is 144."
            },
            "init_prompt": "math calculate carefully step by step answer",
            "num_iters": 500,
            "batch_size": 32,
            "topk": 128,
            "patience": 50
        }
    ]
    
    results = []
    
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Running Task: {task['name']}")
        print(f"{'='*50}\n")
        
        # 1. Baseline Generation on Holdout
        print("--- Baseline Generation (Unoptimized) on Holdout ---")
        unoptimized_toks = torch.tensor(optimizer.tokenizer.encode(task['init_prompt'], add_special_tokens=False), device=optimizer.device)
        baseline_gen = optimizer.generate(unoptimized_toks, task['holdout_test']['user_prompt'])
        print(f"User Prompt: {task['holdout_test']['user_prompt']}")
        print(f"Output:\n{baseline_gen}\n")
        
        # 2. Optimization on Dataset
        print(f"--- Optimizing System Prompt on {len(task['dataset'])} examples ---")
        start_time = time.time()
        best_toks, final_loss = optimizer.optimize(
            dataset=task['dataset'],
            init_prompt=task['init_prompt'],
            num_iters=task['num_iters'],
            batch_size=task['batch_size'],
            topk=task['topk'],
            patience=task['patience']
        )
        opt_time = time.time() - start_time
        
        # 3. Optimized Generation on Holdout
        print("\n--- Optimized Generation on Holdout ---")
        optimized_gen = optimizer.generate(best_toks, task['holdout_test']['user_prompt'])
        print(f"Output:\n{optimized_gen}\n")
        
        optimized_prompt_text = optimizer.tokenizer.decode(best_toks)
        
        results.append({
            "task_name": task['name'],
            "final_loss": final_loss,
            "optimized_prompt": optimized_prompt_text,
            "holdout_target": task['holdout_test']['target'],
            "baseline_output": baseline_gen,
            "optimized_output": optimized_gen,
            "time_seconds": opt_time
        })
        
    # Save results to a report file
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nAll experiments complete! Results saved to experiment_results.json")

if __name__ == "__main__":
    run_experiments()
