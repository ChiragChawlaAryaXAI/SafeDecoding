cd /home/jovyan/SafeDecoding/exp

cat > run_missing_defenses.sh << 'EOFSCRIPT'
#!/bin/bash

API_KEY="gsk_B1sIpad3AQwp6fKKi0d0Fx8"

echo "🔥 Running Missing Defense Combinations"
echo "========================================"
echo "Total: 42 runs (6 defenders × 7 attackers)"
echo "Estimated time: 7-10 hours"
echo ""

#==============================================================================
# PPL DEFENSE (7 attackers)
#==============================================================================

echo "🛡️  PPL DEFENSE"
echo "==============="

echo "[1/42] PPL + AdvBench"
python defense.py --model_name llama2 --attacker AdvBench --defender PPL --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[2/42] PPL + SAP30"
python defense.py --model_name llama2 --attacker SAP30 --defender PPL --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[3/42] PPL + HEx-PHI"
python defense.py --model_name llama2 --attacker HEx-PHI --defender PPL --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[4/42] PPL + GCG"
python defense.py --model_name llama2 --attacker GCG --defender PPL --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[5/42] PPL + AutoDAN"
python defense.py --model_name llama2 --attacker AutoDAN --defender PPL --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[6/42] PPL + PAIR"
python defense.py --model_name llama2 --attacker PAIR --defender PPL --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[7/42] PPL + DeepInception"
python defense.py --model_name llama2 --attacker DeepInception --defender PPL --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

#==============================================================================
# SELF-EXAM DEFENSE (7 attackers)
#==============================================================================

echo ""
echo "🛡️  SELF-EXAM DEFENSE"
echo "===================="

echo "[8/42] Self-Exam + AdvBench"
python defense.py --model_name llama2 --attacker AdvBench --defender Self-Exam --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[9/42] Self-Exam + SAP30"
python defense.py --model_name llama2 --attacker SAP30 --defender Self-Exam --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[10/42] Self-Exam + HEx-PHI"
python defense.py --model_name llama2 --attacker HEx-PHI --defender Self-Exam --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[11/42] Self-Exam + GCG"
python defense.py --model_name llama2 --attacker GCG --defender Self-Exam --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[12/42] Self-Exam + AutoDAN"
python defense.py --model_name llama2 --attacker AutoDAN --defender Self-Exam --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[13/42] Self-Exam + PAIR"
python defense.py --model_name llama2 --attacker PAIR --defender Self-Exam --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[14/42] Self-Exam + DeepInception"
python defense.py --model_name llama2 --attacker DeepInception --defender Self-Exam --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

#==============================================================================
# PARAPHRASE DEFENSE (7 attackers)
#==============================================================================

echo ""
echo "🛡️  PARAPHRASE DEFENSE"
echo "====================="

echo "[15/42] Paraphrase + AdvBench"
python defense.py --model_name llama2 --attacker AdvBench --defender Paraphrase --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[16/42] Paraphrase + SAP30"
python defense.py --model_name llama2 --attacker SAP30 --defender Paraphrase --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[17/42] Paraphrase + HEx-PHI"
python defense.py --model_name llama2 --attacker HEx-PHI --defender Paraphrase --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[18/42] Paraphrase + GCG"
python defense.py --model_name llama2 --attacker GCG --defender Paraphrase --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[19/42] Paraphrase + AutoDAN"
python defense.py --model_name llama2 --attacker AutoDAN --defender Paraphrase --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[20/42] Paraphrase + PAIR"
python defense.py --model_name llama2 --attacker PAIR --defender Paraphrase --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

echo "[21/42] Paraphrase + DeepInception"
python defense.py --model_name llama2 --attacker DeepInception --defender Paraphrase --disable_GPT_judge --device 0 --GPT_API "$API_KEY"

#==============================================================================
# RETOKENIZATION DEFENSE (7 attackers)
#==============================================================================

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

chmod +x remain_run.sh