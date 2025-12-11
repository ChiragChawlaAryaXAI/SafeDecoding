import pandas as pd
import json
from pathlib import Path
from datasets import load_dataset

print("📥 Downloading HEx-PHI CSVs from HuggingFace...\n")

all_prompts = []

# Category mapping
category_info = {
    "category_1": "Illegal Activity",
    "category_3": "Hate/Harass/Violence",
    "category_4": "Malware",
    "category_5": "Physical Harm",
    "category_6": "Economic Harm",
    "category_7": "Fraud/Deception",
    "category_8": "Adult Content",
    "category_9": "Political Campaigning",
    "category_10": "Privacy Violation",
    "category_11": "Tailored Financial Advice"
}

# Download and cache the files
from huggingface_hub import hf_hub_download

for cat_name, cat_label in category_info.items():
    try:
        print(f"📂 Processing {cat_name} ({cat_label})...")
        
        # Download CSV file
        csv_path = hf_hub_download(
            repo_id="LLM-Tuning-Safety/HEx-PHI",
            filename=f"{cat_name}.csv",
            repo_type="dataset"
        )
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        print(f"   Columns: {list(df.columns)}")
        print(f"   Rows: {len(df)}")
        
        # Extract prompts from all columns (they use column names as prompts!)
        for col in df.columns:
            # The column name itself is the prompt
            if len(col) > 10:  # Valid prompt
                all_prompts.append({
                    "id": len(all_prompts),
                    "goal": col.strip(),
                    "category": cat_label,
                    "source": cat_name,
                    "target": ""
                })
        
        # Also check if there are values in cells
        for idx, row in df.iterrows():
            for val in row.values:
                if isinstance(val, str) and len(val) > 10 and not pd.isna(val):
                    all_prompts.append({
                        "id": len(all_prompts),
                        "goal": val.strip(),
                        "category": cat_label,
                        "source": cat_name,
                        "target": ""
                    })
        
        print(f"   ✅ Added prompts from {cat_name}\n")
    
    except Exception as e:
        print(f"   ⚠️  Error with {cat_name}: {e}\n")
        continue

# Remove duplicates
print(f"📊 Total prompts before deduplication: {len(all_prompts)}")
seen = set()
unique_prompts = []
for prompt in all_prompts:
    if prompt['goal'] not in seen:
        seen.add(prompt['goal'])
        prompt['id'] = len(unique_prompts)  # Renumber
        unique_prompts.append(prompt)

all_prompts = unique_prompts

# Save
if all_prompts:
    output_file = "hexphi.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Created {output_file} with {len(all_prompts)} unique prompts")
    
    # Breakdown
    print(f"\n📊 Breakdown by category:")
    from collections import Counter
    category_counts = Counter(p['category'] for p in all_prompts)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {count}")
    
    # Sample
    print(f"\n📝 Sample prompts:")
    for i in range(min(3, len(all_prompts))):
        print(f"\n{i+1}. Category: {all_prompts[i]['category']}")
        print(f"   Goal: {all_prompts[i]['goal'][:100]}...")
else:
    print("\n❌ No prompts extracted!")
