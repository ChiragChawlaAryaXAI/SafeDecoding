#!/usr/bin/env python3
"""
Verify all other datasets: NicheHazardQA, TechHazardQA, HarmfulQA
"""

from datasets import load_dataset

def get_question(sample):
    for key in ["question", "Question", "prompt", "query"]:
        if key in sample:
            return sample[key]
    return ""

def verify_niche_hazard_qa():
    try:
        print("🔍 NicheHazardQA")
        ds = load_dataset("SoftMINER-Group/NicheHazardQA", split="train")
        print(f"  Samples: {len(ds)}")
        print(f"  Columns: {list(ds.column_names)}")
        sample = ds[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample question: {get_question(sample)[:100]}...")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def verify_tech_hazard_qa():
    try:
        print("\n🔍 TechHazardQA")
        ds = load_dataset("SoftMINER-Group/TechHazardQA", split="train")
        print(f"  Samples: {len(ds)}")
        print(f"  Columns: {list(ds.column_names)}")
        sample = ds[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample question: {get_question(sample)[:100]}...")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def verify_harmful_qa():
    try:
        print("\n🔍 HarmfulQA")
        ds = load_dataset("declare-lab/HarmfulQA", split="train")
        print(f"  Samples: {len(ds)}")
        print(f"  Columns: {list(ds.column_names)}")
        sample = ds[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample question: {get_question(sample)[:100]}...")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

if __name__ == "__main__":
    verify_niche_hazard_qa()
    verify_tech_hazard_qa()
    verify_harmful_qa()
