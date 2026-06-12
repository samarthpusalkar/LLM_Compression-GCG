import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

def get_nonascii_toks(tokenizer, device='cpu'):
    """Return a list of non-ascii tokens to filter out, useful for some tokenizers."""
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
        
    return torch.tensor(ascii_toks, device=device)

def token_gradients(model, input_ids, prompt_slice, target_slice, embed_weights):
    """
    Computes gradients of the loss with respect to the one-hot vectors of the prompt tokens.
    """
    # Create one-hot vectors for input_ids
    one_hot = torch.zeros(
        input_ids.shape[0],
        input_ids.shape[1],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        2, 
        input_ids.unsqueeze(2),
        torch.ones(one_hot.shape, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    
    # input_embeds = one_hot @ embed_weights
    input_embeds = torch.matmul(one_hot, embed_weights)
    
    # Forward pass
    outputs = model(inputs_embeds=input_embeds)
    logits = outputs.logits
    
    # Calculate loss on target part only
    # logits shape: [batch, seq_len, vocab_size]
    # We want to predict target_slice. The target tokens are at input_ids[:, target_slice]
    # The logits that predict these tokens are at logits[:, target_slice.start-1 : target_slice.stop-1]
    
    shift_logits = logits[:, target_slice.start-1 : target_slice.stop-1, :].contiguous()
    shift_labels = input_ids[:, target_slice].contiguous()
    
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Backward pass to get gradient w.r.t. one-hot vectors
    loss.backward()
    
    return one_hot.grad.clone()

def sample_control(control_toks, grad, batch_size, topk=256, temp=1.0, not_allowed_tokens=None):
    """
    Randomly sample token substitutions from the top-k negative gradients.
    """
    # grad shape: [1, seq_len, vocab_size]
    # control_toks shape: [control_length]
    
    control_grad = grad[0, :len(control_toks), :]
    
    # We want to minimize loss, so we look at the negative gradient.
    # A negative gradient means increasing that coordinate (setting the one-hot to 1) 
    # will decrease the loss.
    
    if not_allowed_tokens is not None:
        control_grad[:, not_allowed_tokens.to(control_grad.device)] = np.infty
        
    top_indices = (-control_grad).topk(topk, dim=1).indices
    
    control_toks = control_toks.to(control_grad.device)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=control_grad.device
    ).type(torch.int64)
    
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1), device=control_grad.device)
    )
    
    new_control_toks = control_toks.unsqueeze(0).repeat(batch_size, 1)
    new_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    
    return new_control_toks

def optimize_prompt(model_name="gpt2", target_text="The secret passcode is omega protocol nine.", init_prompt="! ! ! ! ! ! ! ! ! !", num_iters=300, batch_size=128, topk=128):
    print(f"Loading model: {model_name}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoModelForCausalLM.from_pretrained(model_name) # to download
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    embed_weights = model.get_input_embeddings().weight.detach()
    
    target_toks = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    control_toks = tokenizer(init_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    
    # Some tokenizers might need an initial space or something, but we'll keep it simple
    not_allowed_tokens = None # get_nonascii_toks(tokenizer, device) # Optional filter
    
    print(f"Initial prompt: '{tokenizer.decode(control_toks)}'")
    print(f"Target text: '{tokenizer.decode(target_toks)}'")
    
    best_loss = float('inf')
    best_control_toks = control_toks.clone()
    
    pbar = tqdm(range(num_iters))
    for i in pbar:
        # 1. Compute gradients for the current control tokens
        input_ids = torch.cat([best_control_toks, target_toks]).unsqueeze(0)
        
        prompt_slice = slice(0, len(best_control_toks))
        target_slice = slice(len(best_control_toks), len(best_control_toks) + len(target_toks))
        
        grad = token_gradients(model, input_ids, prompt_slice, target_slice, embed_weights)
        
        # 2. Sample candidate replacements
        with torch.no_grad():
            new_control_candidates = sample_control(best_control_toks, grad, batch_size, topk, not_allowed_tokens=not_allowed_tokens)
            
            # Evaluate all candidates
            # To save memory, evaluate in smaller batches if necessary, but 64 on small model is fine
            cand_input_ids = torch.cat([
                new_control_candidates, 
                target_toks.unsqueeze(0).repeat(batch_size, 1)
            ], dim=1)
            
            # We can compute loss for all candidates using standard forward pass (no one-hot needed)
            outputs = model(input_ids=cand_input_ids)
            logits = outputs.logits
            
            # target_slice is still the same relative to the concatenated length
            shift_logits = logits[:, target_slice.start-1 : target_slice.stop-1, :].contiguous()
            shift_labels = cand_input_ids[:, target_slice].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses = losses.view(batch_size, -1).mean(dim=1)
            
            min_idx = losses.argmin()
            current_loss = losses[min_idx].item()
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_control_toks = new_control_candidates[min_idx]
                
            pbar.set_description(f"Loss: {best_loss:.4f} | Prompt: {repr(tokenizer.decode(best_control_toks))}")
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print("\nOptimization Complete.")
    print(f"Final Optimized Prompt: {repr(tokenizer.decode(best_control_toks))}")
    print(f"Final Loss: {best_loss:.4f}")
    
    # Generation test
    print("\n--- Generation Test ---")
    input_ids = best_control_toks.unsqueeze(0)
    output = model.generate(input_ids, max_new_tokens=len(target_toks) + 5, do_sample=False)
    generated_text = tokenizer.decode(output[0])
    print(f"Generated text with optimized prompt:\n{generated_text}")

if __name__ == "__main__":
    optimize_prompt()
