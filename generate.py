import os
import torch
import tiktoken
from gpt import GPTLanguageModel

# 1. Setup Tokenizer and Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Construct the absolute path to the weights in the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_dir, 'gpt_rope_weights.pth')

# 2. Load the RoPE Model
model = GPTLanguageModel()
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
m = model.to(device)
m.eval() # Switch to testing mode

# 3. The Prompt
start_text = "Tell me your name" # Change this to whatever you want

# 4. Encode and Generate
print(f"--- Prompting with: '{start_text}' ---")
context = torch.tensor(encode(start_text), dtype=torch.long, device=device).unsqueeze(0)
generated_tokens = m.generate(context, max_new_tokens=200)[0].tolist()

print("\n--- Output ---")
print(decode(generated_tokens))