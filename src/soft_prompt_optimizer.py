import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

def optimize_soft_prompt(model_name="gpt2", target_text="The secret passcode is omega protocol nine.", num_virtual_tokens=5, epochs=300, lr=0.03):
    print(f"Loading model: {model_name}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_size = model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size
    
    # Initialize soft prompt
    init_embeddings = torch.randn(num_virtual_tokens, hidden_size, device=device) * 0.02
    soft_prompt = nn.Parameter(init_embeddings)
    
    optimizer = AdamW([soft_prompt], lr=lr)
    
    # Tokenize target
    inputs = tokenizer(target_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    labels = input_ids.clone()
    
    print(f"Target text: '{target_text}'")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        target_embeds = model.get_input_embeddings()(input_ids)
        
        prompt_embeds = soft_prompt.unsqueeze(0)
        inputs_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
        
        prompt_labels = torch.full((1, num_virtual_tokens), -100, dtype=torch.long, device=device)
        full_labels = torch.cat([prompt_labels, labels], dim=1)
        
        outputs = model(inputs_embeds=inputs_embeds, labels=full_labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    print("\nOptimization Complete.")
    print(f"Final Loss: {loss.item():.4f}")
    
    # Generation test using continuous prompt
    print("\n--- Generation Test (Continuous) ---")
    prompt_embeds = soft_prompt.unsqueeze(0)
    generated_ids = model.generate(
        inputs_embeds=prompt_embeds,
        max_new_tokens=len(input_ids[0]) + 5,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: '{generated_text}'")

    # Projection test
    vocab_embeds = model.get_input_embeddings().weight.detach()
    vocab_norm = vocab_embeds / vocab_embeds.norm(dim=-1, keepdim=True)
    prompt_norm = soft_prompt / soft_prompt.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(prompt_norm, vocab_norm.t())
    best_token_ids = similarity.argmax(dim=-1)
    
    discrete_text = tokenizer.decode(best_token_ids)
    
    with open("results.txt", "w") as f:
        f.write(f"Optimization Complete.\n")
        f.write(f"Final Loss: {loss.item():.4f}\n")
        f.write(f"\n--- Generation Test (Continuous) ---\n")
        f.write(f"Generated text: '{generated_text}'\n")
        f.write(f"\n--- Discrete Projection ---\n")
        f.write(f"Nearest tokens text: '{discrete_text}'\n")
        
    print(f"\n--- Discrete Projection ---")
    print(f"Nearest tokens text: '{discrete_text}'")

if __name__ == "__main__":
    optimize_soft_prompt()
