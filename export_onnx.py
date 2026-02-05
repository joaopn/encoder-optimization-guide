#!/usr/bin/env python3
"""
Export and benchmark ONNX FP16 optimized models for HuggingFace.

This script:
1. Exports a HuggingFace model to ONNX with FP16 optimization
2. Benchmarks it against the original PyTorch model
3. Optionally uploads the optimized model to HuggingFace Hub
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import AutoOptimizationConfig


def get_probabilities(logits, problem_type):
    """Get probabilities from logits based on problem type."""
    if problem_type == 'multi_label_classification':
        return torch.sigmoid(logits)
    else:
        return torch.softmax(logits, dim=-1)


def export_to_onnx(model_id, save_dir, disable_shape_inference=False):
    """Export model to ONNX with FP16 optimization."""
    print(f"Exporting model '{model_id}' to ONNX...")
    
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Export the base model to ONNX
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Setup the Optimizer
    optimizer = ORTOptimizer.from_pretrained(model)

    # 3. Apply O4 Optimization (GPU-only FP16)
    optimization_config = AutoOptimizationConfig.O4(disable_shape_inference=disable_shape_inference)

    print(f"Optimizing model with FP16 (O4 config)...")
    if disable_shape_inference:
        print("  (shape inference disabled)")
    optimizer.optimize(
        save_dir=save_dir,
        optimization_config=optimization_config
    )

    # Rename model_optimized.onnx to model.onnx (standard naming)
    optimized_path = Path(save_dir) / "model_optimized.onnx"
    standard_path = Path(save_dir) / "model.onnx"
    optimized_data_path = Path(save_dir) / "model_optimized.onnx.data"
    standard_data_path = Path(save_dir) / "model.onnx.data"
    
    if optimized_path.exists():
        # Load the ONNX model to update internal references
        model_proto = onnx.load(str(optimized_path), load_external_data=False)
        
        # Update external data references in the model
        if optimized_data_path.exists():
            for tensor in model_proto.graph.initializer:
                if tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL:
                    for i, ext_data in enumerate(tensor.external_data):
                        if ext_data.key == 'location' and ext_data.value == 'model_optimized.onnx.data':
                            tensor.external_data[i].value = 'model.onnx.data'
        
        # Save the updated model with new name
        onnx.save(model_proto, str(standard_path))
        
        # Remove old model file
        optimized_path.unlink()
        
        # Rename the .data file if it exists
        if optimized_data_path.exists():
            optimized_data_path.rename(standard_data_path)

    # 4. Save tokenizer for a complete package
    tokenizer.save_pretrained(save_dir)
    
    print(f"✓ Model exported and optimized to: {save_dir}")


def benchmark_model(model_id, onnx_path, batch_size=1):
    """Benchmark ONNX model against PyTorch version."""
    csv_url = 'https://github.com/joaopn/gpu_benchmark_goemotions/raw/main/data/random_sample_10k.csv.gz'
    field_name = 'body'
    
    # 1. Load Dataset
    print(f"Loading test dataset from {csv_url}...")
    df = pd.read_csv(csv_url)
    total_samples = len(df)
    print(f"Loaded {total_samples} samples.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. Load Models
    print("Loading PyTorch model (reference)...")
    pt_model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True).to("cuda")
    
    print("Loading ONNX model (test)...")
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        onnx_path, 
        file_name="model.onnx", 
        provider="CUDAExecutionProvider",
        use_io_binding=True
    )
    
    # Detect problem type for correct probability calculation
    problem_type = getattr(pt_model.config, "problem_type", "single_label_classification")
    print(f"Detected problem type: {problem_type} (Using {'Sigmoid' if problem_type == 'multi_label_classification' else 'Softmax'})")

    # 3. Batch Processing
    all_diffs = []
    results_meta = []

    print(f"Running inference with batch_size={batch_size}...")
    for start_idx in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_texts = df[field_name].iloc[start_idx:end_idx].tolist()
        batch_indices = range(start_idx, end_idx)

        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")

        with torch.no_grad():
            # PyTorch Inference
            pt_logits = pt_model(**inputs).logits
            pt_probs = get_probabilities(pt_logits, problem_type).detach().cpu().numpy()

            # ONNX Inference
            onnx_logits = onnx_model(**inputs).logits
            onnx_probs = get_probabilities(onnx_logits, problem_type).detach().cpu().numpy()

        # Calculate Absolute Probability Difference
        batch_diffs = np.abs(pt_probs - onnx_probs)
        
        # Store results
        for i, idx in enumerate(batch_indices):
            max_diff_for_sample = np.max(batch_diffs[i])
            
            results_meta.append({
                "index": idx,
                "text_snippet": batch_texts[i][:100],
                "max_diff": max_diff_for_sample,
                "pt_probs": pt_probs[i],
                "onnx_probs": onnx_probs[i]
            })
            all_diffs.append(max_diff_for_sample)

    # 4. Global Statistics
    all_diffs = np.array(all_diffs)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS: PROBABILITY DIFFERENCE STATISTICS")
    print("="*60)
    print(f"Mean:      {np.mean(all_diffs):.8f}")
    print(f"Std Dev:   {np.std(all_diffs):.8f}")
    print(f"Min:       {np.min(all_diffs):.8f}")
    print(f"Max:       {np.max(all_diffs):.8f}")
    print(f"Median:    {np.median(all_diffs):.8f}")
    print("\nQuantiles:")
    quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for q in quantiles:
        print(f"  {int(q*100):>3}th percentile: {np.quantile(all_diffs, q):.8f}")
    
    # 5. Top 3 Worst Offenders
    print("\n" + "="*60)
    print("TOP 3 SAMPLES WITH HIGHEST PROBABILITY DRIFT")
    print("="*60)
    
    sorted_results = sorted(results_meta, key=lambda x: x['max_diff'], reverse=True)
    top_3 = sorted_results[:3]
    id2label = pt_model.config.id2label

    for rank, item in enumerate(top_3, 1):
        print(f"\n#{rank} | Index: {item['index']} | Max Drift: {item['max_diff']:.6f}")
        print(f"Text: \"{item['text_snippet']}...\"")
        print("-" * 65)
        print(f"{'Class':<20} | {'PyTorch %':<10} | {'ONNX %':<10} | {'Diff'}")
        print("-" * 65)
        
        pt_p = item['pt_probs']
        onnx_p = item['onnx_probs']
        
        for class_id, label in id2label.items():
            diff = abs(pt_p[class_id] - onnx_p[class_id])
            marker = "<<" if diff > 0.01 else ""
            print(f"{label[:20]:<20} | {pt_p[class_id]:.6f}   | {onnx_p[class_id]:.6f}   | {diff:.6f} {marker}")
    
    print("\n" + "="*60)
    
    return {
        'mean': np.mean(all_diffs),
        'std': np.std(all_diffs),
        'min': np.min(all_diffs),
        'max': np.max(all_diffs),
        'median': np.median(all_diffs),
        'quantiles': {f'p{int(q*100)}': np.quantile(all_diffs, q) for q in quantiles}
    }


def generate_readme(model_id, stats, disable_shape_inference=False):
    """Generate README content for the model."""
    author_model = model_id
    
    # Format statistics
    stats_text = f"""Mean: {stats['mean']:.8f}
Std Dev: {stats['std']:.8f}
Min: {stats['min']:.8f}
Max: {stats['max']:.8f}
Median: {stats['median']:.8f}

Quantiles:
  25th percentile: {stats['quantiles']['p25']:.8f}
  50th percentile: {stats['quantiles']['p50']:.8f}
  75th percentile: {stats['quantiles']['p75']:.8f}
  90th percentile: {stats['quantiles']['p90']:.8f}
  95th percentile: {stats['quantiles']['p95']:.8f}
  99th percentile: {stats['quantiles']['p99']:.8f}"""
    
    # Generate optimization config line
    if disable_shape_inference:
        opt_config_line = "optimization_config = AutoOptimizationConfig.O4()"
        opt_note = "\n# Note: This model was optimized with disable_shape_inference=True for large model compatibility"
    else:
        opt_config_line = "optimization_config = AutoOptimizationConfig.O4()"
        opt_note = ""
    
    readme = f"""---
tags:
- onnx
- fp16
- optimized
base_model: {author_model}
---

# FP16 Optimized ONNX Model

This model is a ONNX-FP16 optimized version of [{author_model}](https://huggingface.co/{author_model}). It runs exclusively on the GPU. Depending on the model, ONNX-FP16 versions can be 2-3X faster than base PyTorch models. 

For more information on ONNX-FP16 benchmarks vs ONNX and pytorch, as well as the scripts used to generate and check the accuracy of this model, please check https://github.com/joaopn/encoder-optimization-guide. 

On a test set of 10000 reddit comments, the label probability differences between it and the FP32 model were

```
{stats_text}
```

## Usage

The model was generated with:

```python
from optimum.onnxruntime import ORTOptimizer, ORTModelForSequenceClassification, AutoOptimizationConfig
from transformers import AutoTokenizer

model_id = "{author_model}"
save_dir = "./model-onnx-fp16"

# 1. Export the base model to ONNX
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Setup the Optimizer
optimizer = ORTOptimizer.from_pretrained(model)

# 3. Apply O4 Optimization (GPU-only FP16){opt_note}
{opt_config_line}

optimizer.optimize(
    save_dir=save_dir,
    optimization_config=optimization_config
)

# 4. Save tokenizer for a complete package
tokenizer.save_pretrained(save_dir)
```

You will need the GPU version of the ONNX Runtime. It can be installed with:

```bash
pip install optimum[onnxruntime-gpu] --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

For convenience, you can use this [this environment.yml](https://raw.githubusercontent.com/joaopn/encoder-optimization-guide/refs/heads/main/environment.yml) file to create a conda env with all the requirements. Below is an optimized, batched usage example:

```python
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

def sentiment_analysis_batched(df, batch_size, field_name):
    # Replace with your HuggingFace username/model_id after uploading
    model_id = 'YOUR_USERNAME/YOUR_MODEL_ID'
    file_name = 'model.onnx'
    gpu_id = 0
    
    model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name, provider="CUDAExecutionProvider", provider_options={{'device_id': gpu_id}})
    device = torch.device(f"cuda:{{gpu_id}}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    results = []

    # Precompute id2label mapping
    id2label = model.config.id2label

    total_samples = len(df)
    with tqdm(total=total_samples, desc="Processing samples") as pbar:
        for start_idx in range(0, total_samples, batch_size):
            end_idx = start_idx + batch_size
            texts = df[field_name].iloc[start_idx:end_idx].tolist()

            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits)  # Use sigmoid for multi-label classification

            # Collect predictions on GPU
            results.append(predictions)

            pbar.update(end_idx - start_idx)

    # Concatenate all results on GPU
    all_predictions = torch.cat(results, dim=0).cpu().numpy()

    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions, columns=[id2label[i] for i in range(all_predictions.shape[1])])

    # Add prediction columns to the original DataFrame
    combined_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

    return combined_df

df = pd.read_csv('https://github.com/joaopn/gpu_benchmark_goemotions/raw/main/data/random_sample_10k.csv.gz')
df = sentiment_analysis_batched(df, batch_size=8, field_name='body')
```
"""
    return readme


def upload_to_huggingface(save_dir, model_id, hf_token):
    """Upload the optimized model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    
    # Login to HuggingFace
    print("Logging in to HuggingFace...")
    login(token=hf_token)
    
    # Upload to Hub
    print("Uploading model to HuggingFace Hub...")
    api = HfApi()
    
    # Get username
    username = api.whoami()['name']
    
    # Extract model name from original model_id to create a sensible repo name
    model_name = model_id.split('/')[-1]
    repo_name = f"{model_name}-onnx-fp16"
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creating repository: {repo_id}")
    
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        print("✓ Repository created/verified")
        
        print("Uploading files...")
        api.upload_folder(
            folder_path=save_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✓ Model uploaded successfully to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export and benchmark HuggingFace models to ONNX FP16 format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model ID (e.g., 'textdetox/xlmr-large-toxicity-classifier-v2')"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the optimized model (default: ./{model_name}-onnx-fp16)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking (default: 1)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (required for upload)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the upload prompt and don't upload to HuggingFace"
    )
    parser.add_argument(
        "--disable-shape-inference",
        action="store_true",
        help="Disable shape inference during optimization (recommended for very large models to reduce memory usage and optimization time)"
    )
    
    args = parser.parse_args()
    
    # Set default save directory
    if args.save_dir is None:
        model_name = args.model_id.split('/')[-1]
        args.save_dir = f"./{model_name}-onnx-fp16"
    
    print("="*60)
    print("ONNX FP16 MODEL EXPORT AND BENCHMARK")
    print("="*60)
    print(f"Model ID:              {args.model_id}")
    print(f"Save Dir:              {args.save_dir}")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Disable Shape Inference: {args.disable_shape_inference}")
    print("="*60 + "\n")
    
    # Step 1: Export to ONNX
    export_to_onnx(args.model_id, args.save_dir, args.disable_shape_inference)
    
    # Step 2: Benchmark
    print("\n" + "="*60)
    print("STARTING BENCHMARK")
    print("="*60 + "\n")
    stats = benchmark_model(args.model_id, args.save_dir, args.batch_size)
    
    # Step 3: Generate and save README
    print("\nGenerating README...")
    readme_content = generate_readme(args.model_id, stats, args.disable_shape_inference)
    readme_path = Path(args.save_dir) / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ README saved to: {readme_path}")
    
    # Step 4: Upload to HuggingFace
    if not args.no_upload:
        print("\n" + "="*60)
        response = input("Would you like to upload this model to HuggingFace? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            if args.hf_token is None:
                args.hf_token = input("Please enter your HuggingFace API token: ").strip()
                if not args.hf_token:
                    print("No token provided. Skipping upload.")
                    return
            
            upload_to_huggingface(args.save_dir, args.model_id, args.hf_token)
        else:
            print("Upload skipped.")
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
