"""
Local LLM Inference & Quantization Benchmarker
Profiles performance of different quantization levels of Qwen2.5-0.5B-Instruct locally.
"""

import os
import sys
import json
import time
import requests
import threading
import psutil
import pandas as pd
from huggingface_hub import hf_hub_download

# Define constants
OLLAMA_API_URL = "http://localhost:11434"
REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
JUDGE_MODEL = "llama3.1:latest"

# Define the models and their configurations
MODEL_CONFIGS = {
    "qwen2.5-0.5b-q4": {
        "gguf_filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "quantization": "4-bit (Q4_K_M)",
        "ollama_name": "qwen2.5-0.5b-instruct-q4_k_m"
    },
    "qwen2.5-0.5b-q8": {
        "gguf_filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
        "quantization": "8-bit (Q8_0)",
        "ollama_name": "qwen2.5-0.5b-instruct-q8_0"
    },
    "qwen2.5-0.5b-f16": {
        "gguf_filename": "qwen2.5-0.5b-instruct-fp16.gguf",
        "quantization": "16-bit (FP16)",
        "ollama_name": "qwen2.5-0.5b-instruct-fp16"
    }
}

# Standardized prompt dataset
PROMPT_DATASET = [
    {
        "id": 1,
        "category": "Reasoning",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain step-by-step."
    },
    {
        "id": 2,
        "category": "Coding",
        "prompt": "Write a clean Python function to check if a binary tree is balanced. Include type hints and docstring."
    },
    {
        "id": 3,
        "category": "Creative Writing",
        "prompt": "Write a 3-stanza poem about a local LLM running in a dark datacenter, fighting against running out of memory."
    },
    {
        "id": 4,
        "category": "Summarization",
        "prompt": (
            "Summarize the following passage in one concise paragraph:\n"
            "Large language models (LLMs) have revolutionized natural language processing, but their deployment in "
            "resource-constrained environments presents significant challenges. FP16 models require substantial "
            "memory, making local execution on standard consumer devices difficult or impossible. Quantization "
            "techniques, such as 4-bit and 8-bit mapping of model weights, reduce the memory footprint by compressing "
            "weights into lower-precision formats. This optimization enables local LLMs to run efficiently on unified "
            "memory architectures like Apple Silicon, though it introduces a trade-off between model accuracy, response "
            "quality, and execution speed."
        )
    },
    {
        "id": 5,
        "category": "Math",
        "prompt": "Factorize the quadratic expression: 2x^2 + 11x + 15. Show each step clearly."
    }
]

# Helper to check if Ollama is running
def check_ollama_running():
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Download model GGUF from Hugging Face
def download_gguf(filename):
    data_dir = os.path.abspath("data")
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, filename)

    if os.path.exists(target_path):
        print(f"[HF Download] File {filename} already exists at {target_path}. Skipping download.")
        return target_path

    print(f"[HF Download] Downloading {filename} from {REPO_ID} to {target_path}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=data_dir
        )
        print(f"[HF Download] Successfully downloaded {filename} to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"[HF Download] Error downloading {filename}: {e}")
        sys.exit(1)

# Create Ollama model from GGUF using Ollama CLI for robustness
def create_ollama_model(model_name, gguf_path):
    print(f"[Ollama] Registering model {model_name} in Ollama using {gguf_path}...")
    
    # Write a temporary Modelfile
    modelfile_path = "temp_Modelfile"
    with open(modelfile_path, "w") as f:
        # Use quotes for safety with spaces in path
        f.write(f'FROM "{gguf_path}"\n')
    
    try:
        # Run the CLI command
        import subprocess
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"[Ollama] Model {model_name} successfully registered via CLI.")
    except subprocess.CalledProcessError as e:
        print(f"[Ollama Error] Failed to create model {model_name}: {e.stderr}")
        sys.exit(1)
    finally:
        if os.path.exists(modelfile_path):
            os.remove(modelfile_path)

# Get current model RAM/VRAM footprint from Ollama
def get_ollama_loaded_model_vram(model_name):
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/ps")
        if response.status_code == 200:
            data = response.json()
            for m in data.get("models", []):
                # Ollama lists model names either exactly or with :latest tag
                if m.get("name") == model_name or m.get("model") == model_name or m.get("name").startswith(model_name):
                    return m.get("size_vram", 0)
    except Exception:
        pass
    return 0

# Resource monitor class to track peaks during inference
class SystemResourceMonitor(threading.Thread):
    def __init__(self, model_name, interval=0.05):
        super().__init__()
        self.model_name = model_name
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_vram = 0
        self.peak_ram_mb = 0
        self.peak_cpu = 0
        self.initial_ram_mb = psutil.virtual_memory().used / (1024 * 1024)

    def run(self):
        while not self.stop_event.is_set():
            # Get VRAM footprint via Ollama API
            vram = get_ollama_loaded_model_vram(self.model_name)
            vram_mb = vram / (1024 * 1024)
            if vram_mb > self.peak_vram:
                self.peak_vram = vram_mb

            # Get overall system RAM utilization
            current_ram_mb = psutil.virtual_memory().used / (1024 * 1024)
            ram_delta = current_ram_mb - self.initial_ram_mb
            # We track absolute system RAM used and delta RAM used
            if current_ram_mb > self.peak_ram_mb:
                self.peak_ram_mb = current_ram_mb

            # CPU percentage (system-wide)
            cpu = psutil.cpu_percent()
            if cpu > self.peak_cpu:
                self.peak_cpu = cpu

            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join()

# Judge LLM evaluator
def evaluate_response_quality(prompt, response_text):
    print(f"[Judge] Requesting evaluation from {JUDGE_MODEL}...")
    judge_prompt = f"""You are an expert LLM evaluation judge.
Evaluate the following response generated by a local assistant model for the given prompt.

[User Prompt]
{prompt}

[Model Response]
{response_text}

Rate the quality of the response from 1 to 10 based on these criteria:
- **Accuracy**: Is it factually correct and free of hallucination?
- **Formatting**: Is it well-formatted (e.g. correct code blocks, bullet points, headers)?
- **Completeness**: Does it fully answer the prompt?
- **Clarity**: Is it easy to read and understand?

Provide your evaluation as a JSON object with exactly two keys: "score" (integer 1-10) and "explanation" (brief, single-sentence justification).
Do not output any markdown code blocks, just raw JSON.
"""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": JUDGE_MODEL,
                "prompt": judge_prompt,
                "stream": False,
                "format": "json"
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            eval_data = json.loads(result.get("response", "{}"))
            return int(eval_data.get("score", 5)), eval_data.get("explanation", "No explanation provided.")
    except Exception as e:
        print(f"[Judge Warning] Could not score response using local judge: {e}")
    
    # Fallback heuristics if Judge LLM fails or is not present
    word_count = len(response_text.split())
    if word_count < 10:
        return 2, "Response too short."
    elif "error" in response_text.lower():
        return 3, "Response contains errors."
    else:
        return 7, "Quality estimated via heuristics (Judge LLM failed/unavailable)."

# Main benchmarking flow
def main():
    print("====================================================")
    print("Local LLM Inference & Quantization Benchmarker")
    print("====================================================")

    # 1. Check Ollama connection
    if not check_ollama_running():
        print("[Error] Ollama is not running. Please launch Ollama and run this script again.")
        sys.exit(1)
    print("[System] Ollama is running successfully.")

    # 2. Download and register models
    registered_models = []
    for cfg_id, cfg in MODEL_CONFIGS.items():
        # Download GGUF
        gguf_path = download_gguf(cfg["gguf_filename"])
        # Register in Ollama
        create_ollama_model(cfg["ollama_name"], gguf_path)
        registered_models.append(cfg["ollama_name"])

    print("[System] All models downloaded and registered.")
    
    # 3. Setup folders
    results_dir = os.path.abspath("web/results")
    os.makedirs(results_dir, exist_ok=True)

    benchmark_data = {}

    # 4. Benchmarking models
    for cfg_id, cfg in MODEL_CONFIGS.items():
        model_name = cfg["ollama_name"]
        print(f"\n====================================================")
        print(f"Benchmarking Model: {model_name} ({cfg['quantization']})")
        print("====================================================")

        # Pre-warm model to load it into unified memory (ensures accurate timing)
        print("[Warmup] Sending pre-warm inference request...")
        try:
            requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={"model": model_name, "prompt": "Pre-warm test. Reply with one word: ready.", "stream": False},
                timeout=30
            )
            # Sleep briefly to let stats settle
            time.sleep(2)
        except Exception as e:
            print(f"[Warning] Warmup request failed: {e}")

        model_runs = []

        for p_idx, p_data in enumerate(PROMPT_DATASET):
            print(f"\n[Prompt {p_data['id']}/{len(PROMPT_DATASET)}] Category: {p_data['category']}")
            print(f"Prompt: {p_data['prompt'][:60]}...")

            # Start resource monitor thread
            monitor = SystemResourceMonitor(model_name)
            monitor.start()

            # Execute API call
            start_time = time.time()
            try:
                response = requests.post(
                    f"{OLLAMA_API_URL}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": p_data["prompt"],
                        "stream": False
                    },
                    timeout=60
                )
                end_time = time.time()
                monitor.stop()

                if response.status_code == 200:
                    api_data = response.json()
                    response_text = api_data.get("response", "")
                    
                    # Timing extraction from API metadata
                    total_dur_ns = api_data.get("total_duration", 0)
                    load_dur_ns = api_data.get("load_duration", 0)
                    prompt_eval_count = api_data.get("prompt_eval_count", 0)
                    prompt_eval_ns = api_data.get("prompt_eval_duration", 1)
                    eval_count = api_data.get("eval_count", 0)
                    eval_ns = api_data.get("eval_duration", 1)

                    # Compute speeds
                    ttft_sec = prompt_eval_ns / 1e9
                    tokens_per_sec = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0
                    generation_dur_sec = eval_ns / 1e9

                    print(f"  Metrics:")
                    print(f"    - TTFT: {ttft_sec:.3f} s")
                    print(f"    - Generation Speed: {tokens_per_sec:.2f} tokens/s")
                    print(f"    - Generated Tokens: {eval_count}")
                    print(f"    - Peak VRAM footprint: {monitor.peak_vram:.2f} MB")
                    print(f"    - Peak Host RAM: {monitor.peak_ram_mb:.2f} MB")
                    print(f"    - Peak CPU: {monitor.peak_cpu:.1f}%")

                    # Judge score
                    judge_score, judge_exp = evaluate_response_quality(p_data["prompt"], response_text)
                    print(f"    - Judge Score: {judge_score}/10 ({judge_exp[:50]}...)")

                    # Log prompt details
                    model_runs.append({
                        "prompt_id": p_data["id"],
                        "category": p_data["category"],
                        "prompt": p_data["prompt"],
                        "response": response_text,
                        "ttft_sec": round(ttft_sec, 4),
                        "tokens_per_sec": round(tokens_per_sec, 2),
                        "eval_count": eval_count,
                        "generation_time_sec": round(generation_dur_sec, 4),
                        "peak_vram_mb": round(monitor.peak_vram, 2),
                        "peak_ram_mb": round(monitor.peak_ram_mb, 2),
                        "peak_cpu_percent": round(monitor.peak_cpu, 1),
                        "judge_score": judge_score,
                        "judge_explanation": judge_exp
                    })
                else:
                    print(f"  [Error] API returned status code {response.status_code}")
            except Exception as e:
                monitor.stop()
                print(f"  [Error] Failed to benchmark prompt: {e}")

            # Cooldown sleep between prompts
            time.sleep(2)

        # Compute summary metrics for this quantization
        if model_runs:
            df = pd.DataFrame(model_runs)
            summary = {
                "avg_ttft_sec": round(float(df["ttft_sec"].mean()), 4),
                "avg_tokens_per_sec": round(float(df["tokens_per_sec"].mean()), 2),
                "avg_judge_score": round(float(df["judge_score"].mean()), 2),
                "avg_peak_vram_mb": round(float(df["peak_vram_mb"].mean()), 2),
                "avg_peak_ram_mb": round(float(df["peak_ram_mb"].mean()), 2),
                "avg_peak_cpu_percent": round(float(df["peak_cpu_percent"].mean()), 1)
            }
        else:
            summary = {}

        benchmark_data[cfg_id] = {
            "model_name": model_name,
            "quantization": cfg["quantization"],
            "summary": summary,
            "runs": model_runs
        }

        # Unload the model from VRAM/RAM to ensure clean baseline for next model
        print(f"[Ollama] Unloading model {model_name} to reset memory footprint...")
        try:
            # Setting keep_alive to 0 immediately unloads the model in Ollama
            requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": 0},
                timeout=5
            )
            time.sleep(3)
        except Exception:
            pass

    # 5. Export JSON results
    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_data, f, indent=4)
    print(f"\n[System] Exported detailed benchmark data to {json_path}")

    # 6. Export flattened CSV results for Pandas/plotting convenience
    flat_rows = []
    for cfg_id, cfg_data in benchmark_data.items():
        for run in cfg_data["runs"]:
            flat_rows.append({
                "model_id": cfg_id,
                "model_name": cfg_data["model_name"],
                "quantization": cfg_data["quantization"],
                "prompt_id": run["prompt_id"],
                "category": run["category"],
                "ttft_sec": run["ttft_sec"],
                "tokens_per_sec": run["tokens_per_sec"],
                "eval_count": run["eval_count"],
                "generation_time_sec": run["generation_time_sec"],
                "peak_vram_mb": run["peak_vram_mb"],
                "peak_ram_mb": run["peak_ram_mb"],
                "peak_cpu_percent": run["peak_cpu_percent"],
                "judge_score": run["judge_score"]
            })
    
    if flat_rows:
        csv_path = os.path.join(results_dir, "benchmark_results.csv")
        df_flat = pd.DataFrame(flat_rows)
        df_flat.to_csv(csv_path, index=False)
        print(f"[System] Exported flattened CSV metrics to {csv_path}")

        # 7. Generate static diagnostic PNG charts
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Speed comparison
            sns.barplot(ax=axes[0], data=df_flat, x="quantization", y="tokens_per_sec", ci=None, palette="viridis")
            axes[0].set_title("Inference Speed (Tokens/Sec) - Higher is Better")
            axes[0].set_ylabel("Speed (tokens/s)")
            axes[0].set_xlabel("Quantization Level")

            # Memory Footprint comparison
            sns.barplot(ax=axes[1], data=df_flat, x="quantization", y="peak_vram_mb", ci=None, palette="magma")
            axes[1].set_title("VRAM Allocation (MB) - Lower is Better")
            axes[1].set_ylabel("VRAM Footprint (MB)")
            axes[1].set_xlabel("Quantization Level")

            # Judge Score comparison
            sns.barplot(ax=axes[2], data=df_flat, x="quantization", y="judge_score", ci=None, palette="rocket")
            axes[2].set_title("Response Quality (Score 1-10) - Higher is Better")
            axes[2].set_ylabel("Average Judge Score")
            axes[2].set_xlabel("Quantization Level")

            plt.tight_layout()
            chart_path = os.path.join(results_dir, "benchmark_summary.png")
            plt.savefig(chart_path, dpi=150)
            print(f"[System] Generated benchmark summary visualization at {chart_path}")
        except Exception as e:
            print(f"[System Warning] Could not generate matplotlib charts: {e}")

    print("\n====================================================")
    print("Benchmark completed successfully!")
    print("Run `run.sh` to spin up the dashboard and view metrics.")
    print("====================================================")

if __name__ == "__main__":
    main()
