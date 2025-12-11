import json
import os
from pathlib import Path

# Path to SAP30 dataset
sap30_path = Path("SAP/datasets/SAP30")

all_behaviors = []
behavior_id = 0

# Actual categories from ls output
categories = [
    "fraud", "politics", "pornography_sexual_minors", "race", 
    "religion", "suicide", "terrorism", "violence"
]

print(f"📂 Processing SAP30 from: {sap30_path}\n")

# Process each category
for category in categories:
    category_path = sap30_path / category / "generated_cases.json"
    
    if not category_path.exists():
        print(f"❌ Skipping {category} - file not found")
        continue
    
    print(f"✅ Processing {category}...")
    
    try:
        with open(category_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count_before = len(all_behaviors)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of items
            for item in data:
                if isinstance(item, dict):
                    # Dictionary with prompt/behavior/goal
                    behavior = None
                    
                    # Try all possible keys
                    for key in ['prompt', 'behavior', 'goal', 'question', 'text', 'instruction', 'query']:
                        if key in item and item[key]:
                            behavior = item[key]
                            break
                    
                    # If no key found, try values
                    if not behavior and item:
                        # Get first non-empty string value
                        for v in item.values():
                            if isinstance(v, str) and len(v) > 10:
                                behavior = v
                                break
                    
                    if behavior:
                        all_behaviors.append({
                            "id": behavior_id,
                            "goal": behavior.strip(),
                            "category": category,
                            "target": ""
                        })
                        behavior_id += 1
                
                elif isinstance(item, str) and len(item) > 10:
                    # Direct string
                    all_behaviors.append({
                        "id": behavior_id,
                        "goal": item.strip(),
                        "category": category,
                        "target": ""
                    })
                    behavior_id += 1
        
        elif isinstance(data, dict):
            # Dictionary structure
            
            # Check if it has a 'cases' or 'behaviors' key
            if 'cases' in data:
                data = data['cases']
            elif 'behaviors' in data:
                data = data['behaviors']
            elif 'prompts' in data:
                data = data['prompts']
            
            # If still dict, iterate
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 10:
                        all_behaviors.append({
                            "id": behavior_id,
                            "goal": value.strip(),
                            "category": category,
                            "target": ""
                        })
                        behavior_id += 1
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and len(item) > 10:
                                all_behaviors.append({
                                    "id": behavior_id,
                                    "goal": item.strip(),
                                    "category": category,
                                    "target": ""
                                })
                                behavior_id += 1
            elif isinstance(data, list):
                # Now it's a list, recurse
                for item in data:
                    if isinstance(item, str) and len(item) > 10:
                        all_behaviors.append({
                            "id": behavior_id,
                            "goal": item.strip(),
                            "category": category,
                            "target": ""
                        })
                        behavior_id += 1
        
        added = len(all_behaviors) - count_before
        print(f"   Added {added} behaviors")
    
    except Exception as e:
        print(f"⚠️  Error processing {category}: {e}")
        import traceback
        traceback.print_exc()

# Save unified SAP30 dataset
if all_behaviors:
    output_file = "sap30.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_behaviors, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Created {output_file} with {len(all_behaviors)} behaviors")
    
    # Breakdown
    print(f"\n📊 Breakdown by category:")
    from collections import Counter
    category_counts = Counter(b['category'] for b in all_behaviors)
    for cat, count in sorted(category_counts.items()):
        print(f"   {cat}: {count}")
    
    # Sample
    print(f"\n📝 Sample behavior:")
    print(f"   Category: {all_behaviors[0]['category']}")
    print(f"   Goal: {all_behaviors[0]['goal'][:100]}...")
else:
    print("\n❌ No behaviors extracted!")
