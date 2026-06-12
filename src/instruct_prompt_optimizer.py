import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

def token_gradients(model, sys_start_toks, control_toks, dataset_toks, embed_weights):
    device = model.device
    
    one_hot_control = torch.zeros(
        1, len(control_toks), embed_weights.shape[0],
        device=device, dtype=embed_weights.dtype
    )
    one_hot_control.scatter_(
        2, 
        control_toks.unsqueeze(0).unsqueeze(2),
        torch.ones(one_hot_control.shape, device=device, dtype=embed_weights.dtype)
    )
    one_hot_control.requires_grad_()
    
    for user_toks, target_toks in dataset_toks:
        control_embeds = torch.matmul(one_hot_control, embed_weights)
        sys_start_embeds = embed_weights[sys_start_toks].unsqueeze(0)
        user_embeds = embed_weights[user_toks].unsqueeze(0)
        target_embeds = embed_weights[target_toks].unsqueeze(0)
        
        inputs_embeds = torch.cat([sys_start_embeds, control_embeds, user_embeds, target_embeds], dim=1)
        
        outputs = model(inputs_embeds=inputs_embeds)
        logits = outputs.logits
        
        target_start_idx = len(sys_start_toks) + len(control_toks) + len(user_toks)
        shift_logits = logits[:, target_start_idx-1 : target_start_idx-1 + len(target_toks), :].contiguous()
        shift_labels = target_toks.unsqueeze(0).contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss = loss / len(dataset_toks)
        loss.backward()
        
    return one_hot_control.grad.clone()

def sample_control(control_toks, grad, batch_size, topk=256):
    control_grad = grad[0]
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

class InstructPromptOptimizer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"Loading model: {model_name}")
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.embed_weights = self.model.get_input_embeddings().weight.detach()
        
        # ChatML templates for Qwen
        self.SYS_START = "<|im_start|>system\n"
        self.SYS_END = "<|im_end|>\n"
        self.USER_START = "<|im_start|>user\n"
        self.USER_END = "<|im_end|>\n"
        self.ASSISTANT_START = "<|im_start|>assistant\n"
        
        self.sys_start_toks = torch.tensor(self.tokenizer.encode(self.SYS_START, add_special_tokens=False), device=self.device)
        self.sys_end_toks = self.tokenizer.encode(self.SYS_END, add_special_tokens=False)
        self.user_start_toks = self.tokenizer.encode(self.USER_START, add_special_tokens=False)
        self.user_end_toks = self.tokenizer.encode(self.USER_END, add_special_tokens=False)
        self.assistant_start_toks = self.tokenizer.encode(self.ASSISTANT_START, add_special_tokens=False)

    def _prepare_dataset(self, dataset):
        dataset_toks = []
        for item in dataset:
            user_msg_toks = self.tokenizer.encode(item['user_prompt'], add_special_tokens=False)
            user_section_toks = torch.tensor(
                self.sys_end_toks + self.user_start_toks + user_msg_toks + self.user_end_toks + self.assistant_start_toks, 
                device=self.device
            )
            target_toks = torch.tensor(self.tokenizer.encode(item['target'], add_special_tokens=False), device=self.device)
            dataset_toks.append((user_section_toks, target_toks))
        return dataset_toks

    def optimize(self, dataset, init_prompt="! ! ! ! ! ! ! ! ! !", num_iters=200, batch_size=64, topk=128, patience=50):
        control_toks = torch.tensor(self.tokenizer.encode(init_prompt, add_special_tokens=False), device=self.device)
        dataset_toks = self._prepare_dataset(dataset)
        
        best_loss = float('inf')
        best_control_toks = control_toks.clone()
        epochs_without_improvement = 0
        
        pbar = tqdm(range(num_iters))
        for i in pbar:
            # 1. Compute averaged gradients across dataset
            grad = token_gradients(self.model, self.sys_start_toks, best_control_toks, dataset_toks, self.embed_weights)
            
            # 2. Sample candidates
            with torch.no_grad():
                new_control_candidates = sample_control(best_control_toks, grad, batch_size, topk)
                
                # 3. Evaluate candidates across the dataset
                candidate_losses = torch.zeros(batch_size, device=self.device)
                
                for user_section_toks, target_toks in dataset_toks:
                    cand_input_ids = torch.cat([
                        self.sys_start_toks.unsqueeze(0).repeat(batch_size, 1),
                        new_control_candidates, 
                        user_section_toks.unsqueeze(0).repeat(batch_size, 1),
                        target_toks.unsqueeze(0).repeat(batch_size, 1)
                    ], dim=1)
                    
                    outputs = self.model(input_ids=cand_input_ids)
                    logits = outputs.logits
                    
                    target_start_idx = len(self.sys_start_toks) + len(best_control_toks) + len(user_section_toks)
                    
                    shift_logits = logits[:, target_start_idx-1 : target_start_idx-1 + len(target_toks), :].contiguous()
                    shift_labels = cand_input_ids[:, target_start_idx : target_start_idx + len(target_toks)].contiguous()
                    
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    # Accumulate average loss for this sample
                    candidate_losses += losses.view(batch_size, -1).mean(dim=1) / len(dataset_toks)
                
                min_idx = candidate_losses.argmin()
                current_loss = candidate_losses[min_idx].item()
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_control_toks = new_control_candidates[min_idx]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    
                pbar.set_description(f"Loss: {best_loss:.4f} | Pat: {epochs_without_improvement}/{patience} | Prompt: {repr(self.tokenizer.decode(best_control_toks))}")
                
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered. No improvement for {patience} epochs.")
                    break
                
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        print("\nOptimization Complete.")
        print(f"Final Optimized Prompt: {repr(self.tokenizer.decode(best_control_toks))}")
        print(f"Final Loss: {best_loss:.4f}")
        
        return best_control_toks, best_loss
        
    def generate(self, control_toks, user_text, max_new_tokens=50):
        user_msg_toks = self.tokenizer.encode(user_text, add_special_tokens=False)
        user_section_toks = torch.tensor(
            self.sys_end_toks + self.user_start_toks + user_msg_toks + self.user_end_toks + self.assistant_start_toks, 
            device=self.device
        )
        
        input_ids = torch.cat([
            self.sys_start_toks,
            control_toks,
            user_section_toks
        ]).unsqueeze(0)
        
        output = self.model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_ids = output[0, input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids)
