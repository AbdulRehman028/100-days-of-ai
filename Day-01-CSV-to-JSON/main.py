import pandas as pd
import sys
import os

def csv_to_json(csv_file, json_file):
    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Convert to JSON (records format is easier for ML/NLP pipelines)
        df.to_json(json_file, orient="records", indent=4)

        print(f"✅ Successfully converted '{csv_file}' → '{json_file}'")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input.csv> <output.json>")
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]

    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

    csv_to_json(csv_file, json_file)