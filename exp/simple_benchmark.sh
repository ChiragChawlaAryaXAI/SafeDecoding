#!/bin/bash

API_KEY="Daaldo"
JUDGE="llama-4-scout"

echo "🔥 Starting Simple Benchmark..."
echo "================================"
echo ""

#==============================================================================
# TABLE 1: JAILBREAK ATTACKS - SafeDecoding Defense
#==============================================================================

echo "📊 TABLE 1: Jailbreak Attacks with SafeDecoding"
echo "================================================"
echo ""

# AdvBench + SafeDecoding
echo "1/14: AdvBench + SafeDecoding"
python defense.py --model_name llama2 --attacker AdvBench --defender SafeDecoding --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# SAP30 + SafeDecoding
echo "2/14: SAP30 + SafeDecoding"
python defense.py --model_name llama2 --attacker SAP30 --defender SafeDecoding --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# HEx-PHI + SafeDecoding
echo "3/14: HEx-PHI + SafeDecoding"
python defense.py --model_name llama2 --attacker HEx-PHI --defender SafeDecoding --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# GCG + SafeDecoding
echo "4/14: GCG + SafeDecoding"
python defense.py --model_name llama2 --attacker GCG --defender SafeDecoding --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# AutoDAN + SafeDecoding
echo "5/14: AutoDAN + SafeDecoding"
python defense.py --model_name llama2 --attacker AutoDAN --defender SafeDecoding --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# PAIR + SafeDecoding
echo "6/14: PAIR + SafeDecoding"
python defense.py --model_name llama2 --attacker PAIR --defender SafeDecoding --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# DeepInception + SafeDecoding
echo "7/14: DeepInception + SafeDecoding"
python defense.py --model_name llama2 --attacker DeepInception --defender SafeDecoding --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

#==============================================================================
# TABLE 1: JAILBREAK ATTACKS - No Defense (Baseline)
#==============================================================================

echo ""
echo "📊 TABLE 1: Jailbreak Attacks WITHOUT Defense (Baseline)"
echo "========================================================="
echo ""

# AdvBench + No Defense
echo "8/14: AdvBench + No Defense"
python defense.py --model_name llama2 --attacker AdvBench --defense_off --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# SAP30 + No Defense
echo "9/14: SAP30 + No Defense"
python defense.py --model_name llama2 --attacker SAP30 --defense_off --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# HEx-PHI + No Defense
echo "10/14: HEx-PHI + No Defense"
python defense.py --model_name llama2 --attacker HEx-PHI --defense_off --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# GCG + No Defense
echo "11/14: GCG + No Defense"
python defense.py --model_name llama2 --attacker GCG --defense_off --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# AutoDAN + No Defense
echo "12/14: AutoDAN + No Defense"
python defense.py --model_name llama2 --attacker AutoDAN --defense_off --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# PAIR + No Defense
echo "13/14: PAIR + No Defense"
python defense.py --model_name llama2 --attacker PAIR --defense_off --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

# DeepInception + No Defense
echo "14/14: DeepInception + No Defense"
python defense.py --model_name llama2 --attacker DeepInception --defense_off --judge_model "$JUDGE" --device 0 --GPT_API "$API_KEY"
sleep 5

#==============================================================================
# SUMMARY
#==============================================================================

echo ""
echo "✅ ALL BENCHMARKS COMPLETE!"
echo "==========================="
echo ""
echo "📂 Results saved in: ../exp_outputs/"
echo ""
echo "📊 To view summary, run:"
echo "   python analyze_results.py"
echo ""

