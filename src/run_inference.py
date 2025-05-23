#!/usr/bin/env python
import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append("src")
from BOND import helpsteer2_prompt2messages

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Llama model")
    parser.add_argument("--model_path", type=str, default="pareto_distilled_llama", help="Path to the distilled model")
    parser.add_argument("--output_file", type=str, default="eval_responses.jsonl", help="Path to save generated responses")
    parser.add_argument("--eval_dataset", type=str, default="nvidia/HelpSteer2", help="Evaluation dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}")
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"  # For better generation
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    
    # Load evaluation dataset
    if args.eval_dataset.endswith(".jsonl"):
        # Load from local JSONL file
        print(f"Loading dataset from local file: {args.eval_dataset}")
        eval_ds = load_dataset("json", data_files=args.eval_dataset, split=args.split)
    else:
        # Load from Hugging Face datasets
        print(f"Loading dataset from Hugging Face: {args.eval_dataset}")
        eval_ds = load_dataset(args.eval_dataset, split=args.split)
    
    # Check dataset format
    has_prompt_messages = "prompt_messages" in eval_ds.column_names
    has_prompt = "prompt" in eval_ds.column_names
    
    # Process prompts and generate responses
    results = []
    
    for i in tqdm(range(0, len(eval_ds), args.batch_size), desc="Generating responses"):
        batch = eval_ds[i:i+args.batch_size]
        
        # Convert prompts to messages format
        batch_messages = []
        for item in batch:
            try:
                if has_prompt_messages:
                    # Already in the right format
                    messages = item["prompt_messages"]
                elif has_prompt:
                    # Text prompt needs conversion
                    if isinstance(item["prompt"], list):
                        # Already a list of message dicts
                        messages = item["prompt"]
                    else:
                        # String prompt needs parsing
                        messages = helpsteer2_prompt2messages(item["prompt"])
                else:
                    # No recognized format, use the whole item as user message
                    messages = [{"role": "user", "content": str(item)}]
                
                batch_messages.append(messages)
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                # Fallback: treat as single user message
                content = str(item.get("prompt", item))[:1000]  # Limit length for safety
                batch_messages.append([{"role": "user", "content": content}])
        
        # Format inputs using the chat template
        formatted_prompts = []
        for messages in batch_messages:
            # Format as a chat (without the assistant's response)
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=True  # Important for inference
            )
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode generated outputs
        for j, output in enumerate(outputs):
            # Get only the generated text (not the prompt)
            prompt_length = inputs.input_ids.shape[1]
            response = tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
            
            # Save in the desired format
            idx = i + j
            if idx < len(eval_ds):
                entry = {
                    "prompt": batch_messages[j],
                    "response": response,
                    "id": f"eval_{idx}"
                }
                results.append(entry)
    
    # Save responses to jsonl file
    print(f"Saving {len(results)} responses to {args.output_file}")
    with open(args.output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main() 