#!/usr/bin/env python3
import sys
import os
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from safe_eval_openrouter import DictJudge, GPTJudge_OpenRouter

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate with OpenRouter")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="meta-llama/llama-3.3-70b-instruct:free")
    parser.add_argument("--GPT_API", type=str, required=True)
    parser.add_argument("--disable_GPT_judge", action="store_true")
    parser.add_argument("--multi_processing", type=int, default=20)
    return parser.parse_args()

def main():
    args = get_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logging.info(f"📂 Loading: {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'data' in data:
        results = data['data']
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError("Unknown input file format")
    
    logging.info(f"✅ Loaded {len(results)} responses")
    
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
        logging.error("❌ No valid goal-response pairs found!")
        return
    
    logging.info("🔍 Running Dictionary Judge...")
    dict_judge = DictJudge()
    scores_dict = dict_judge.eval_batch(responses)
    
    scores_gpt = None
    reasons = None
    
    if not args.disable_GPT_judge:
        logging.info(f"🔥 Running GPT Judge with: {args.judge_model}...")
        gpt_judge = GPTJudge_OpenRouter('gpt', mp=args.multi_processing, 
                                        judge_model=args.judge_model, api=args.GPT_API)
        
        goals_responses_pairs = [(goals[i], responses[i]) for i in range(len(goals))]
        scores, reasons = gpt_judge.evaluate(goals_responses_pairs)
        scores_gpt = gpt_judge.score_parse(scores)
    
    eval_results = []
    for i in range(len(goals)):
        eval_results.append({
            "id": i,
            "goal": goals[i],
            "instruction": instructions[i],
            "output": responses[i],
            "dic_score": scores_dict[i],
            "GPT_score": scores_gpt[i] if scores_gpt else None,
            "GPT_reason": reasons[i] if reasons else None
        })
    
    gpt_score_sum = sum([r['GPT_score'] for r in eval_results if r['GPT_score'] is not None])
    defense_success_count = sum([1 for r in eval_results if r['dic_score'] == 1 or r['dic_score'] == True])
    
    harmful_score = gpt_score_sum / len(eval_results) if scores_gpt else 0
    asr = 100 - (defense_success_count / len(eval_results)) * 100
    
    logging.info("")
    logging.info("="*60)
    logging.info("📊 EVALUATION RESULTS")
    logging.info("="*60)
    logging.info(f"Total samples: {len(eval_results)}")
    logging.info(f"Harmful Score: {harmful_score:.2f}")
    logging.info(f"ASR: {asr:.2f}%")
    logging.info(f"Defense Success: {defense_success_count}/{len(eval_results)}")
    logging.info("="*60)
    
    if args.output_file:
        output_path = args.output_file
    else:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_openrouter_eval.json"
    
    with open(output_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    logging.info(f"💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()