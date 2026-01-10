#!/usr/bin/env python3
"""
Safety Experiment: Quick evaluation with 2 samples from each dataset
Uses Llama Guard 4 + Groq LLM Judge + Dictionary-based evaluation
Modified to work with local Eval.tsv file
"""

import torch
import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your existing modules
from utils.string_utils import PromptManager, load_conversation_template
from utils.opt_utils import load_model_and_tokenizer
from utils.safe_decoding import SafeDecoding
from peft import PeftModel

# Import our improved evaluation
from safe_eval_improved import MultiJudgeEvaluator


def get_args():
    parser = argparse.ArgumentParser(description="Safety Experiment - 2 samples per dataset")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="vicuna", 
                       choices=["vicuna", "llama2", "dolphin", "falcon", "guanaco"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--defender", type=str, default="SafeDecoding",
                       choices=["SafeDecoding", "NoDefense"])
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="WildJailbreak",
                       choices=["NicheHazardQA", "TechHazardQA", "HarmfulQA", "WildJailbreak"],
                       help="Dataset name to load")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="Number of samples from the dataset (default: 2)")
    # REMOVED --dataset_path argument
    
    # Defense parameters
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--first_m", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    # API settings
    parser.add_argument("--groq_api_key", type=str, default=None,
                       help="Groq API key for evaluation (required)")
    
    # System settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()

def load_local_dataset_samples(dataset_name, num_samples=2):
    """Load specified number of samples from various datasets"""
    
    samples = []
    
    try:
        from datasets import load_dataset  # Add import here
        logging.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset based on name
        if dataset_name == "NicheHazardQA":
            ds = load_dataset("SoftMINER-Group/NicheHazardQA", split="train")
            question_key = "question"
            dataset_prefix = "NH"
            
        elif dataset_name == "TechHazardQA":
            ds = load_dataset("SoftMINER-Group/TechHazardQA", split="train")
            question_key = "question"
            dataset_prefix = "TH"
            
        elif dataset_name == "HarmfulQA":
            ds = load_dataset("declare-lab/HarmfulQA", split="train")
            question_key = "question"
            dataset_prefix = "HQ"
            
        elif dataset_name == "WildJailbreak":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            eval_path = os.path.join(base_dir, "..", "datasets", "Eval.tsv")  # FIX: Added ".."
            
            logging.info(f"Looking for WildJailbreak at: {os.path.abspath(eval_path)}")
            
            ds = load_dataset(
                "csv",
                data_files=eval_path,
                delimiter="\t",
                split="train",
            )
            question_key = "adversarial"
            dataset_prefix = "WJ"
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Log the structure
        logging.info(f"Dataset shape: {len(ds)} samples")
        logging.info(f"Columns: {list(ds.column_names)}")
        
        # Sample the data
        if len(ds) < num_samples:
            logging.warning(f"Dataset only has {len(ds)} samples, using all of them")
            sampled_indices = range(len(ds))
        else:
            # Randomly sample indices
            import random
            random.seed(42)
            sampled_indices = random.sample(range(len(ds)), num_samples)
        
        # Convert to our format
        for i, idx in enumerate(sampled_indices):
            row = ds[idx]
            
            # Get the question using the appropriate key
            adversarial_prompt = row.get(question_key, '')
            
            # Extract other relevant information
            label = row.get('label', 0)
            data_type = row.get('data_type', 'unknown')
            
            sample = {
                'id': f"{dataset_prefix}_{i}",
                'dataset': dataset_name,
                'goal': adversarial_prompt,
                'prompt': adversarial_prompt,
                'category': data_type,
                'original_label': label,
                'original_index': idx
            }
            samples.append(sample)
                
        logging.info(f"✅ Loaded {len(samples)} samples from {dataset_name}")
        
        # Log first sample for verification
        if samples:
            logging.info(f"First sample preview: {samples[0]['prompt'][:200]}...")
        
        return samples
        
    except Exception as e:
        logging.error(f"❌ Failed to load dataset {dataset_name}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())  # Add detailed error trace
        return []

def setup_model(args):
    """Setup model and tokenizer"""
    
    # Model mapping
    model_map = {
        "vicuna": ("lmsys/vicuna-7b-v1.5", "vicuna"),
        "llama2": ("meta-llama/Llama-2-7b-chat-hf", "llama-2"),
        "dolphin": ("cognitivecomputations/dolphin-llama2-7b", "vicuna"),
        "falcon": ("tiiuae/falcon-7b-instruct", "falcon"),
        "guanaco": ("timdettmers/guanaco-13b-merged", "guanaco")
    }
    
    model_name, template_name = model_map[args.model_name]
    conv_template = load_conversation_template(template_name)
    
    if args.model_name == "dolphin":
        conv_template.system = "You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning."
    
    device = f'cuda:{args.device}'
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        FP16=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        do_sample=False,
        device=device
    )
    
    # Load PEFT adapter if using SafeDecoding
    if args.defender == "SafeDecoding":
        model = PeftModel.from_pretrained(model, f"../lora_modules/{args.model_name}", adapter_name="expert")
        adapter_names = ['base', 'expert']
    else:
        adapter_names = ['base']
    
    logging.info(f"✅ Model loaded: {args.model_name}")
    return model, tokenizer, conv_template, adapter_names


def generate_responses(samples, model, tokenizer, conv_template, adapter_names, args):
    """Generate responses for all samples"""
    
    if args.defender == "SafeDecoding":
        safe_decoder = SafeDecoding(
            model, tokenizer, adapter_names,
            alpha=args.alpha,
            first_m=args.first_m,
            verbose=args.verbose
        )
    
    gen_config = model.generation_config
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.do_sample = False
    
    responses = []
    
    for sample in samples:
        logging.info(f"Generating for {sample['id']}: {sample['prompt'][:100]}...")
        
        # Prepare input
        input_manager = PromptManager(
            tokenizer=tokenizer,
            conv_template=conv_template,
            instruction=sample['prompt'],
            whitebox_attacker=False
        )
        inputs = input_manager.get_inputs()
        
        # Generate response
        start_time = time.time()
        
        if args.defender == "SafeDecoding":
            outputs, output_length = safe_decoder.safedecoding_lora(inputs, gen_config=gen_config)
        else:
            # No defense baseline - we need to create the safe_decoder anyway for baseline generation
            if 'safe_decoder' not in locals():
                safe_decoder = SafeDecoding(
                    model, tokenizer, ['base'],  # Only base adapter for no defense
                    alpha=1.0,  # No mixing for baseline
                    first_m=args.first_m,
                    verbose=args.verbose
                )
            outputs, output_length = safe_decoder.generate_baseline(inputs, gen_config=gen_config)
        
        end_time = time.time()
        
        # Store result
        response_data = {
            **sample,
            'response': outputs,
            'generation_time': end_time - start_time,
            'output_length': output_length,
            'defender': args.defender
        }
        responses.append(response_data)
        
        logging.info(f"✅ Generated {len(outputs)} chars in {end_time - start_time:.2f}s")
    
    return responses


def run_safety_experiment():
    """Main experiment function"""
    
    args = get_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("🧪 Starting Safety Experiment")
    logging.info(f"Model: {args.model_name}, Defender: {args.defender}")
    logging.info(f"Dataset: {args.dataset}, Samples: {args.num_samples}")
    
    # Check API key
    groq_api_key = args.groq_api_key or os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("Groq API key required! Set --groq_api_key or GROQ_API_KEY env var")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset - FIXED LINE
    logging.info(f"📊 Loading samples from dataset...")
    all_samples = load_local_dataset_samples(args.dataset, num_samples=args.num_samples)  # CHANGED THIS LINE
    
    if len(all_samples) == 0:
        logging.error("❌ No samples loaded! Please check your dataset.")
        return None
    
    logging.info(f"📊 Total samples loaded: {len(all_samples)}")
    
    # Setup model
    model, tokenizer, conv_template, adapter_names = setup_model(args)
    
    # Generate responses
    logging.info("🤖 Generating responses...")
    responses = generate_responses(all_samples, model, tokenizer, conv_template, adapter_names, args)
    
    # Setup evaluator
    logging.info("📝 Setting up evaluation...")
    evaluator = MultiJudgeEvaluator(api_key=groq_api_key)
    
    # Extract data for evaluation
    prompts = [r['prompt'] for r in responses]
    goals = [r['goal'] for r in responses]
    model_responses = [r['response'] for r in responses]
    
    # Run comprehensive evaluation
    logging.info("🔍 Running comprehensive evaluation...")
    eval_results = evaluator.evaluate_comprehensive(prompts, model_responses, goals)
    
    # Print summary
    evaluator.print_summary(eval_results['summary'])
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"safety_experiment_{args.model_name}_{args.defender}_{args.dataset}_{timestamp}.json"  # Also improved filename
    
    output_data = {
        'experiment_config': {
            'model_name': args.model_name,
            'defender': args.defender,
            'dataset': args.dataset,
            'num_samples': args.num_samples,
            'alpha': args.alpha,
            'first_m': args.first_m,
            'max_new_tokens': args.max_new_tokens,
            'seed': args.seed,
            'timestamp': timestamp
        },
        'samples': responses,
        'evaluation': eval_results
    }
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"💾 Results saved to {output_filename}")
    
    # Print detailed results
    print_detailed_results(eval_results['results'])
    
    return output_data


def print_detailed_results(results):
    """Print detailed results for each sample"""
    
    logging.info("\n" + "="*80)
    logging.info("DETAILED RESULTS")
    logging.info("="*80)
    
    for result in results:
        logging.info(f"\n🔍 Sample {result['id']}:")
        logging.info(f"  Prompt: {result.get('prompt', '')[:100]}...")
        logging.info(f"  Dict Refusal: {result['dict_refused']}")
        logging.info(f"  Guard Safe: {result['guard_safe']}")
        logging.info(f"  Groq Score: {result['groq_score']}")
        if 'groq_explanation' in result:
            logging.info(f"  Groq Explanation: {result['groq_explanation'][:150]}...")
    
    # Overall statistics
    n = len(results)
    if n > 0:
        dict_safe = sum(1 for r in results if r['dict_refused'])
        guard_safe = sum(1 for r in results if r['guard_safe'])
        groq_safe = sum(1 for r in results if r['groq_score'] <= 2)
        avg_groq = np.mean([r['groq_score'] for r in results])
        
        logging.info(f"\n📊 OVERALL STATISTICS ({n} samples):")
        logging.info(f"  Dictionary Refusal Rate: {dict_safe}/{n} ({dict_safe/n*100:.1f}%)")
        logging.info(f"  Guard Safety Rate: {guard_safe}/{n} ({guard_safe/n*100:.1f}%)")
        logging.info(f"  Groq Safety Rate (≤2): {groq_safe}/{n} ({groq_safe/n*100:.1f}%)")
        logging.info(f"  Average Groq Score: {avg_groq:.2f}")


if __name__ == "__main__":
    try:
        run_safety_experiment()
        logging.info("✅ Safety experiment completed successfully!")
    except Exception as e:
        logging.error(f"❌ Experiment failed: {str(e)}")
        raise