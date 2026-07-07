import os
import sys

repo_path = "/Users/kartik/Documents/ANEForge-main"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import aneforge as af
from transformers import AutoTokenizer

def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading {model_name} on the Apple Neural Engine via ANEForge...")
    
    try:
        model = af.load_llm(model_name, compress="fp32")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("You might need to authenticate with Hugging Face using 'huggingface-cli login'.")
        return 1

    # The ANE requires a fixed, pre-allocated memory size (MAX_LEN) before running.
    # We set it very high (2048) so it won't cut off. The model will automatically 
    # stop generating as soon as it's finished (when it emits its EOS token).
    MAX_LEN = 2048
    print(f"Warming up ANE (compiling the graph for {MAX_LEN} tokens)...")
    model.warmup(MAX_LEN)
    
    messages = []
    print("\nReady! Type a message (Ctrl-D or 'exit' to quit).")
    
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print()
            break
            
        if not prompt: 
            continue
        if prompt.lower() in ("exit", "quit"): 
            break
            
        messages.append({"role": "user", "content": prompt})
        
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt_text)
        except Exception:
            ids = tokenizer.encode(prompt)
            
        if len(ids) >= MAX_LEN - 8:
            print("ane> (Context limit reached. Please restart the script to clear memory.)\n")
            messages.pop() # remove the last message so it doesn't get stuck
            continue
            
        print("ane> ", end="", flush=True)
        
        generated_tokens = []
        def token_callback(t):
            generated_tokens.append(t)
            print(tokenizer.decode([t]), end="", flush=True)

        model.generate(
            ids, 
            max_new_tokens=MAX_LEN - len(ids) - 1, 
            max_len=MAX_LEN, 
            eos_id=tokenizer.eos_token_id,
            on_token=token_callback
        )
        print("\n")
        
        reply_text = tokenizer.decode(generated_tokens)
        messages.append({"role": "assistant", "content": reply_text})
        
    return 0

if __name__ == "__main__":
    sys.exit(main())