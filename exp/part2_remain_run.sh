#!/bin/bash

API_KEY=""

echo "🔥 Running Missing Defense Combinations"
echo "========================================"
echo "Total: 42 runs (6 defenders × 7 attackers)"
echo "Estimated time: 7-10 hours"
echo ""



echo ""
echo "🛡️  RETOKENIZATION DEFENSE"
echo "=========================="

echo "[22/42] Retokenization + AdvBench"
python defense.py --model_name llama2 --attacker AdvBench --defender Retokenization --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[23/42] Retokenization + SAP30"
python defense.py --model_name llama2 --attacker SAP30 --defender Retokenization --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[24/42] Retokenization + HEx-PHI"
python defense.py --model_name llama2 --attacker HEx-PHI --defender Retokenization --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[25/42] Retokenization + GCG"
python defense.py --model_name llama2 --attacker GCG --defender Retokenization --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[26/42] Retokenization + AutoDAN"
python defense.py --model_name llama2 --attacker AutoDAN --defender Retokenization --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[27/42] Retokenization + PAIR"
python defense.py --model_name llama2 --attacker PAIR --defender Retokenization --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[28/42] Retokenization + DeepInception"
python defense.py --model_name llama2 --attacker DeepInception --defender Retokenization --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

#==============================================================================
# SELF-REMINDER DEFENSE (7 attackers)
#==============================================================================

echo ""
echo "🛡️  SELF-REMINDER DEFENSE"
echo "========================="

echo "[29/42] Self-Reminder + AdvBench"
python defense.py --model_name llama2 --attacker AdvBench --defender Self-Reminder --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[30/42] Self-Reminder + SAP30"
python defense.py --model_name llama2 --attacker SAP30 --defender Self-Reminder --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[31/42] Self-Reminder + HEx-PHI"
python defense.py --model_name llama2 --attacker HEx-PHI --defender Self-Reminder --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[32/42] Self-Reminder + GCG"
python defense.py --model_name llama2 --attacker GCG --defender Self-Reminder --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[33/42] Self-Reminder + AutoDAN"
python defense.py --model_name llama2 --attacker AutoDAN --defender Self-Reminder --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[34/42] Self-Reminder + PAIR"
python defense.py --model_name llama2 --attacker PAIR --defender Self-Reminder --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[35/42] Self-Reminder + DeepInception"
python defense.py --model_name llama2 --attacker DeepInception --defender Self-Reminder --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

#==============================================================================
# ICD DEFENSE (7 attackers)
#==============================================================================

echo ""
echo "🛡️  ICD DEFENSE"
echo "==============="

echo "[36/42] ICD + AdvBench"
python defense.py --model_name llama2 --attacker AdvBench --defender ICD --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[37/42] ICD + SAP30"
python defense.py --model_name llama2 --attacker SAP30 --defender ICD --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[38/42] ICD + HEx-PHI"
python defense.py --model_name llama2 --attacker HEx-PHI --defender ICD --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[39/42] ICD + GCG"
python defense.py --model_name llama2 --attacker GCG --defender ICD --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[40/42] ICD + AutoDAN"
python defense.py --model_name llama2 --attacker AutoDAN --defender ICD --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[41/42] ICD + PAIR"
python defense.py --model_name llama2 --attacker PAIR --defender ICD --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[42/42] ICD + DeepInception"
python defense.py --model_name llama2 --attacker DeepInception --defender ICD --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

#==============================================================================
# COMPLETION
#==============================================================================

EOFSCRIPT

chmod +x remain_run.sh
echo ""
echo "✅ ALL 42 RUNS COMPLETE!"
echo "========================"
echo ""
echo "📂 Results saved in: ../exp_outputs/"
echo ""
echo "Next steps:"
echo "1. Run batch_eval_all.py to evaluate all results"
echo "2. Check exp_eval folder for evaluation outputs"
echo ""



EOFSCRIPT

chmod +x part2_remain_run.sh
