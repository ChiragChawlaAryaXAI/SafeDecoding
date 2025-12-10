"""
Run MT-Bench evaluation on SafeDecoding models
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Import SafeDecoding adapter
from mtbench_adapter import SafeDecodingModelAdapter


def load_mtbench_questions():
    """Load MT-Bench questions from FastChat"""
    mtbench_path = Path.home() / "FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
    
    if not mtbench_path.exists():
        raise FileNotFoundError(
            f"MT-Bench questions not found at {mtbench_path}\n"
            "Please ensure FastChat is installed at ~/FastChat"
        )
    
    questions = []
    with open(mtbench_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"✅ Loaded {len(questions)} MT-Bench questions")
    return questions


def generate_model_answers(adapter, questions, output_file):
    """Generate answers for all MT-Bench questions"""
    
    answers = []
    
    for question in tqdm(questions, desc="Generating answers"):
        question_id = question['question_id']
        category = question['category']
        turns = question['turns']
        
        # Generate for each turn
        conversation = []
        for turn_idx, turn in enumerate(turns):
            # For turn 2, include previous context
            if turn_idx == 1:
                # Combine turn 1 question + answer with turn 2 question
                context = f"Previous question: {turns[0]}\nYour previous answer: {conversation[0]['content']}\n\nFollow-up question: {turn}"
                response = adapter.generate(context, max_tokens=1024)
            else:
                response = adapter.generate(turn, max_tokens=1024)
            
            conversation.append({
                "role": "assistant",
                "content": response
            })
        
        # Save answer
        answer = {
            "question_id": question_id,
            "answer_id": f"safedecoding_{question_id}",
            "model_id": adapter.model_name + "_safedecoding",
            "choices": [{"index": 0, "turns": [c["content"] for c in conversation]}],
            "tstamp": None
        }
        answers.append(answer)
    
    # Save answers
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for answer in answers:
            f.write(json.dumps(answer) + '\n')
    
    print(f"✅ Saved answers to {output_path}")
    return answers


def run_judgment(model_id, judge_model="gpt-4", api_key=None):
    """Run GPT-4 judgment on generated answers"""
    
    fastchat_path = Path.home() / "FastChat/fastchat/llm_judge"
    
    # Prepare command
    cmd = f"""
    cd {fastchat_path} && python gen_judgment.py \\
        --model-list {model_id} \\
        --judge-model {judge_model} \\
        --mode single
    """
    
    if api_key:
        # Set API key for Fireworks
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_API_BASE'] = 'https://api.fireworks.ai/inference/v1'
    
    print(f"🔥 Running judgment with {judge_model}...")
    print(f"Command: {cmd}")
    
    os.system(cmd)


def show_results(model_id):
    """Display MT-Bench results"""
    
    fastchat_path = Path.home() / "FastChat/fastchat/llm_judge"
    result_file = fastchat_path / f"data/mt_bench/model_judgment/gpt-4_single.jsonl"
    
    if not result_file.exists():
        print(f"❌ Results not found at {result_file}")
        return
    
    # Parse results
    scores = []
    with open(result_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('model') == model_id:
                scores.append(data['score'])
    
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n{'='*50}")
        print(f"MT-Bench Results for {model_id}")
        print(f"{'='*50}")
        print(f"Average Score: {avg_score:.2f}/10")
        print(f"Total Questions: {len(scores)}")
        print(f"{'='*50}\n")
    else:
        print(f"❌ No results found for model: {model_id}")


def main():
    parser = argparse.ArgumentParser(description="Run MT-Bench on SafeDecoding")
    parser.add_argument("--model_name", type=str, default="llama2", choices=["llama2", "vicuna"])
    parser.add_argument("--defender", type=str, default="SafeDecoding")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument("--api_key", type=str, default=None, help="Fireworks API key for judgment")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation, only run judgment")
    parser.add_argument("--skip_judgment", action="store_true", help="Skip judgment, only generate answers")
    
    args = parser.parse_args()
    
    model_id = f"{args.model_name}_{args.defender.lower()}"
    output_dir = f"../exp_outputs/mtbench/{model_id}"
    answer_file = f"{output_dir}/model_answer.jsonl"
    
    # Step 1: Generate answers
    if not args.skip_generation:
        print(f"\n{'='*50}")
        print(f"Step 1: Generating MT-Bench Answers")
        print(f"{'='*50}\n")
        
        # Load adapter
        adapter = SafeDecodingModelAdapter(
            model_name=args.model_name,
            defender=args.defender,
            device=f"cuda:{args.device}",
            alpha=args.alpha
        )
        
        # Load questions
        questions = load_mtbench_questions()
        
        # Generate answers
        generate_model_answers(adapter, questions, answer_file)
        
        # Copy to FastChat location
        fastchat_answer_dir = Path.home() / "FastChat/fastchat/llm_judge/data/mt_bench/model_answer"
        fastchat_answer_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        dest_file = fastchat_answer_dir / f"{model_id}.jsonl"
        shutil.copy(answer_file, dest_file)
        print(f"✅ Copied answers to FastChat: {dest_file}")
    
    # Step 2: Run judgment
    if not args.skip_judgment:
        print(f"\n{'='*50}")
        print(f"Step 2: Running GPT-4 Judgment")
        print(f"{'='*50}\n")
        
        if args.api_key is None:
            print("⚠️  No API key provided. Judgment will fail without API key.")
            print("Use --api_key to provide Fireworks API key")
        else:
            run_judgment(model_id, judge_model="gpt-4", api_key=args.api_key)
    
    # Step 3: Show results
    if not args.skip_generation and not args.skip_judgment:
        print(f"\n{'='*50}")
        print(f"Step 3: Results")
        print(f"{'='*50}\n")
        show_results(model_id)


if __name__ == "__main__":
    main()