#!/usr/bin/env python3
"""
Batch Evaluation Script for SafeDecoding
Evaluates all generated files and saves results in exp_eval folder
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from safe_eval import DictJudge, GPTJudge


JUDGE_MODEL = "llama-4-scout"

# Files to evaluate
FILES_TO_EVAL = [
    #"/home/jovyan/SafeDecoding/exp_outputs/nodefense_llama2_AdvBench_520_2025-12-13 13:05:41/nodefense_llama2_AdvBench_520_2025-12-13 13:05:41.json",
    "/home/jovyan/SafeDecoding/exp_outputs/nodefense_llama2_AutoDAN_50_2025-12-13 15:05:08/nodefense_llama2_AutoDAN_50_2025-12-13 15:05:08.json",
    "/home/jovyan/SafeDecoding/exp_outputs/nodefense_llama2_DeepInception_50_2025-12-13 15:20:03/nodefense_llama2_DeepInception_50_2025-12-13 15:20:03.json",
    "/home/jovyan/SafeDecoding/exp_outputs/nodefense_llama2_GCG_50_2025-12-13 14:59:03/nodefense_llama2_GCG_50_2025-12-13 14:59:03.json",
    "/home/jovyan/SafeDecoding/exp_outputs/nodefense_llama2_HEx-PHI_300_2025-12-13 14:21:19/nodefense_llama2_HEx-PHI_300_2025-12-13 14:21:19.json",
    "/home/jovyan/SafeDecoding/exp_outputs/nodefense_llama2_PAIR_50_2025-12-13 15:10:46/nodefense_llama2_PAIR_50_2025-12-13 15:10:46.json",
    "/home/jovyan/SafeDecoding/exp_outputs/nodefense_llama2_SAP30_240_2025-12-13 13:53:55/nodefense_llama2_SAP30_240_2025-12-13 13:53:55.json",
    #"/home/jovyan/SafeDecoding/exp_outputs/SafeDecoding_llama2_AdvBench_520_2025-12-13 06:24:46/SafeDecoding_llama2_AdvBench_520_2025-12-13 06:24:46.json",
    "/home/jovyan/SafeDecoding/exp_outputs/SafeDecoding_llama2_AutoDAN_50_2025-12-13 12:41:19/SafeDecoding_llama2_AutoDAN_50_2025-12-13 12:41:19.json",
    "/home/jovyan/SafeDecoding/exp_outputs/SafeDecoding_llama2_DeepInception_50_2025-12-13 12:54:26/SafeDecoding_llama2_DeepInception_50_2025-12-13 12:54:26.json",
    "/home/jovyan/SafeDecoding/exp_outputs/SafeDecoding_llama2_GCG_50_2025-12-13 12:35:31/SafeDecoding_llama2_GCG_50_2025-12-13 12:35:31.json",
    "/home/jovyan/SafeDecoding/exp_outputs/SafeDecoding_llama2_HEx-PHI_300_2025-12-13 11:57:41/SafeDecoding_llama2_HEx-PHI_300_2025-12-13 11:57:41.json",
    "/home/jovyan/SafeDecoding/exp_outputs/SafeDecoding_llama2_PAIR_50_2025-12-13 12:47:14/SafeDecoding_llama2_PAIR_50_2025-12-13 12:47:14.json",
    "/home/jovyan/SafeDecoding/exp_outputs/SafeDecoding_llama2_SAP30_240_2025-12-13 11:29:16/SafeDecoding_llama2_SAP30_240_2025-12-13 11:29:16.json",
]

def evaluate_file(input_file, output_dir, judge_model, api_key):
    """Evaluate a single file"""
    
    logging.info(f"\n{'='*70}")
    logging.info(f"📂 Processing: {Path(input_file).name}")
    logging.info(f"{'='*70}")
    
    # Load input file
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"❌ Failed to load {input_file}: {e}")
        return None
    
    # Extract responses
    if isinstance(data, dict) and 'data' in data:
        results = data['data']
    elif isinstance(data, list):
        results = data
    else:
        logging.error(f"❌ Unknown format in {input_file}")
        return None
    
    logging.info(f"✅ Loaded {len(results)} responses")
    
    # Extract goals and responses
    goals = []
    responses = []
    instructions = []
    
    for result in results:
        goal = result.get('goal') or result.get('query') or result.get('instruction')
        response = result.get('output') or result.get('response')
        instruction = result.get('instruction') or goal
        
        if goal and response:
            goals.append(goal)
            responses.append(response)
            instructions.append(instruction)
    
    logging.info(f"📊 Extracted {len(goals)} goal-response pairs")
    
    if len(goals) == 0:
        logging.error("❌ No valid pairs found!")
        return None
    
    # Dictionary Judge
    logging.info("🔍 Running Dictionary Judge...")
    dict_judge = DictJudge()
    scores_dict = dict_judge.eval_batch(responses)
    
    # GPT Judge
    logging.info(f"🔥 Running GPT Judge with {judge_model}...")
    gpt_judge = GPTJudge('gpt', mp=20, judge_model=judge_model, api=api_key)
    
    goals_responses_pairs = [(goals[i], responses[i]) for i in range(len(goals))]
    scores, reasons = gpt_judge.evaluate(goals_responses_pairs)
    scores_gpt = gpt_judge.score_parse(scores)
    
    # Compile results
    eval_results = []
    for i in range(len(goals)):
        output_formatted = {
            "id": i,
            "goal": goals[i],
            "instruction": instructions[i],
            "output": responses[i],
            "dic_score": scores_dict[i],
            "GPT_score": scores_gpt[i],
            "GPT_reason": reasons[i]
        }
        eval_results.append(output_formatted)
    
    # Calculate metrics
    gpt_score_sum = sum([r['GPT_score'] for r in eval_results if r['GPT_score'] is not None])
    defense_success_count = sum([1 for r in eval_results if r['dic_score'] == 1 or r['dic_score'] == True])
    
    harmful_score = gpt_score_sum / len(eval_results)
    asr = 100 - (defense_success_count / len(eval_results)) * 100
    
    logging.info("")
    logging.info("📊 RESULTS:")
    logging.info(f"   Harmful Score: {harmful_score:.2f}")
    logging.info(f"   ASR: {asr:.2f}%")
    logging.info(f"   Defense Success: {defense_success_count}/{len(eval_results)}")
    
    # Save to exp_eval folder
    input_path = Path(input_file)
    filename = input_path.stem
    
    # Create output path in exp_eval
    output_file = output_dir / f"{filename}_safe_eval.json"
    
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    logging.info(f"💾 Saved to: {output_file}")
    
    # Save summary
    summary = {
        "input_file": str(input_file),
        "total_samples": len(eval_results),
        "harmful_score": harmful_score,
        "asr_percentage": asr,
        "defense_success_count": defense_success_count,
        "judge_model": judge_model
    }
    
    summary_file = output_dir / f"{filename}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logging.info(f"📈 Summary saved to: {summary_file}")
    
    return summary

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Create exp_eval directory
    output_dir = Path("/home/jovyan/SafeDecoding/exp_eval")
    output_dir.mkdir(exist_ok=True)
    
    logging.info(f"\n🔥 BATCH EVALUATION STARTED")
    logging.info(f"{'='*70}")
    logging.info(f"Total files: {len(FILES_TO_EVAL)}")
    logging.info(f"Judge model: {JUDGE_MODEL}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"{'='*70}\n")
    
    # Track results
    all_summaries = []
    
    # Evaluate each file
    for i, file_path in enumerate(FILES_TO_EVAL, 1):
        logging.info(f"\n[{i}/{len(FILES_TO_EVAL)}] Processing...")
        
        summary = evaluate_file(file_path, output_dir, JUDGE_MODEL, API_KEY)
        
        if summary:
            all_summaries.append(summary)
        
        logging.info(f"✅ File {i}/{len(FILES_TO_EVAL)} complete\n")
    
    # Save overall summary
    logging.info(f"\n{'='*70}")
    logging.info("📊 BATCH EVALUATION COMPLETE")
    logging.info(f"{'='*70}")
    logging.info(f"Successfully evaluated: {len(all_summaries)}/{len(FILES_TO_EVAL)} files")
    
    overall_summary_file = output_dir / "overall_summary.json"
    with open(overall_summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=4)
    
    logging.info(f"💾 Overall summary saved to: {overall_summary_file}")
    
    # Print comparison table
    print_comparison_table(all_summaries)

def print_comparison_table(summaries):
    """Print Table 1 style comparison"""
    
    from collections import defaultdict
    
    results = defaultdict(lambda: defaultdict(dict))
    
    for summary in summaries:
        filepath = summary['input_file']
        filename = Path(filepath).stem
        parts = filename.split('_')
        
        if len(parts) >= 3:
            defender = parts[0]
            model = parts[1]
            attacker = parts[2]
            
            results[attacker][defender] = {
                'harmful_score': summary['harmful_score'],
                'asr': summary['asr_percentage'],
                'total': summary['total_samples']
            }
    
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS (Table 1 Format)")
    print("="*80)
    
    for attacker in sorted(results.keys()):
        print(f"\n🎯 {attacker}")
        print("-" * 70)
        print(f"{'Defense':<20} {'Harmful Score':<15} {'ASR (%)':<10} {'Samples'}")
        print("-" * 70)
        
        for defender in sorted(results[attacker].keys()):
            metrics = results[attacker][defender]
            print(f"{defender:<20} {metrics['harmful_score']:<15.2f} {metrics['asr']:<10.1f} {metrics['total']}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()