"""
Load SAP30 dataset from GitHub
"""
import json
import pandas as pd
import requests

def download_sap30():
    """Download SAP30 from GitHub"""
    url = "https://raw.githubusercontent.com/Aatrox103/SAP/main/datasets/SAP30/behaviors.csv"
    
    # Download CSV
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse CSV
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    
    # Convert to SafeDecoding format (same as AdvBench)
    sap30_data = []
    for idx, row in df.iterrows():
        sap30_data.append({
            "id": idx,
            "goal": row['Behavior'],
            "category": row.get('Category', 'unknown')
        })
    
    # Save
    output_path = "sap30.json"
    with open(output_path, 'w') as f:
        json.dump(sap30_data, f, indent=4)
    
    print(f"✅ Downloaded {len(sap30_data)} SAP30 behaviors to {output_path}")
    return sap30_data

if __name__ == "__main__":
    download_sap30()