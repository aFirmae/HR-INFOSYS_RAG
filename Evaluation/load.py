import json
import glob
import os

def load_benchmarks(json_folder):
    all_data = []
    for file_path in glob.glob(os.path.join(json_folder, "*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"⚠️ Skipping {file_path}: not a list")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    return all_data

benchmark_data = load_benchmarks(r"D:\AGENTIC_AI\PROJECTS\hr-assistant\Benchmark")
print(f"Loaded {len(benchmark_data)} benchmark queries.")
