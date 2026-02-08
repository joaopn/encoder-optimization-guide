import argparse
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
import time
import os
import multiprocessing as mp

transformers.logging.set_verbosity_error()


def load_model(device_type, model_type, model_id, file_name, gpu_id, num_threads):
    """Load model and return (model, device)."""
    if device_type == 'gpu':
        device = torch.device(f'cuda:{gpu_id}')
    elif device_type == 'cpu':
        device = torch.device('cpu')
        torch.set_num_threads(num_threads)

    if model_type == 'torch':
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(device)
    elif model_type == 'onnx':
        if device_type == 'cpu':
            model = ORTModelForSequenceClassification.from_pretrained(
                model_id, file_name=file_name, provider="CPUExecutionProvider")
        elif device_type == 'gpu':
            model = ORTModelForSequenceClassification.from_pretrained(
                model_id, file_name=file_name, provider="CUDAExecutionProvider",
                provider_options={'device_id': gpu_id})

    return model, device


def pre_tokenize(texts, tokenizer):
    """Tokenize all texts without padding. Returns list of dicts."""
    encodings = tokenizer(texts, truncation=True, max_length=512, padding=False)
    return [{'input_ids': ids, 'attention_mask': mask}
            for ids, mask in zip(encodings['input_ids'], encodings['attention_mask'])]


def prepare_batches(tokenized_data, batch_size, tokenizer):
    """Chunk tokenized data into batches and pad each batch. Returns list of padded tensors."""
    batches = []
    n = len(tokenized_data)
    for start_idx in range(0, n, batch_size):
        batch = tokenized_data[start_idx:start_idx + batch_size]
        batch_enc = {
            'input_ids': [b['input_ids'] for b in batch],
            'attention_mask': [b['attention_mask'] for b in batch],
        }
        padded = tokenizer.pad(batch_enc, return_tensors='pt')
        batches.append(padded)
    return batches


def count_messages(batches):
    """Count total messages across all batches."""
    return sum(b['input_ids'].shape[0] for b in batches)


def inference_loop(model, device, batches):
    """Run inference on pre-padded batches. Only does .to(device) + forward pass."""
    for padded in batches:
        input_ids = padded['input_ids'].to(device)
        attention_mask = padded['attention_mask'].to(device)
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)


def warmup(model, device, batches):
    """Run one batch as warmup (untimed)."""
    padded = batches[0]
    input_ids = padded['input_ids'].to(device)
    attention_mask = padded['attention_mask'].to(device)
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)


def worker_process(worker_id, tokenized_shard, batch_size, model_type, model_id, file_name,
                   device_type, gpu_id, num_threads, barrier, result_queue):
    """Worker process: load model, pad batches, warmup, wait at barrier, run timed inference."""
    model, device = load_model(device_type, model_type, model_id, file_name, gpu_id, num_threads)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Pad batches inside the worker (avoids pickling tensors across processes)
    batches = prepare_batches(tokenized_shard, batch_size, tokenizer)
    n = count_messages(batches)

    warmup(model, device, batches)
    if device_type == 'gpu':
        torch.cuda.synchronize(gpu_id)

    barrier.wait()

    start = time.time()
    inference_loop(model, device, batches)
    if device_type == 'gpu':
        torch.cuda.synchronize(gpu_id)
    elapsed = time.time() - start

    result_queue.put((worker_id, n, elapsed))


def benchmark_multi(tokenized_data, batch_size, num_workers, model_type, model_id, file_name,
                    device_type, gpu_id, num_threads):
    """Multi-worker benchmark. Returns messages/s."""
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(num_workers)
    result_queue = ctx.Queue()

    # Split raw tokenized data (plain lists, no tensors) into per-worker shards
    shard_size = len(tokenized_data) // num_workers
    shards = []
    for i in range(num_workers):
        start = i * shard_size
        end = start + shard_size if i < num_workers - 1 else len(tokenized_data)
        shards.append(tokenized_data[start:end])

    processes = []
    for i, shard in enumerate(shards):
        p = ctx.Process(target=worker_process, args=(
            i, shard, batch_size, model_type, model_id, file_name,
            device_type, gpu_id, num_threads, barrier, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = [result_queue.get() for _ in range(num_workers)]
    total_msgs = sum(r[1] for r in results)
    max_elapsed = max(r[2] for r in results)

    return total_msgs / max_elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GPUs running encoder models")
    parser.add_argument("--dataset", type=str, choices=["normal", "filtered"], default="normal", help="Dataset to use (normal or filtered)")
    parser.add_argument("--model", type=str, choices=["torch", "onnx", "onnx-fp16"], required=True, help="Model to use (torch, onnx or onnx-fp16)")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu", help="Device to use (cpu or gpu)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default 0)")
    parser.add_argument("--batches", type=str, default="1,2,4,8,16,32", help="Comma-separated batch sizes to run")
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads to use (default 1)")
    parser.add_argument("--workers", type=str, default="1", help="Comma-separated worker counts to run (default 1)")
    parser.add_argument("--save", action="store_true", help="Save results to results/<model>_<gpu>.txt")

    args = parser.parse_args()

    # Models
    model_ids = {
        "torch": {'model_type': 'torch', 'model_id': "SamLowe/roberta-base-go_emotions", 'file_name': None},
        "onnx": {'model_type': 'onnx', 'model_id': "SamLowe/roberta-base-go_emotions-onnx", 'file_name': "onnx/model.onnx"},
        "onnx-fp16": {'model_type': 'onnx', 'model_id': "joaopn/roberta-base-go_emotions-onnx-fp16", 'file_name': "model.onnx"}
    }

    if args.model == "onnx-fp16" and args.device == "cpu":
        raise ValueError("ONNX FP16 models are only supported on GPUs")

    field_name = "body"
    model_params = model_ids[args.model]

    if args.dataset == "filtered":
        str_dataset = 'data/random_sample_10k_filtered.csv.gz'
    else:
        str_dataset = 'data/random_sample_10k.csv.gz'

    df = pd.read_csv(str_dataset, compression='gzip')
    texts = df[field_name].tolist()

    # Pre-tokenize dataset once
    print("Pre-tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_params['model_id'])
    tokenized_data = pre_tokenize(texts, tokenizer)
    print(f"Pre-tokenized {len(tokenized_data)} texts")

    batch_sizes = [int(x) for x in args.batches.split(',')]
    worker_counts = [int(x) for x in args.workers.split(',')]

    results = {}

    for num_workers in worker_counts:
        for batch_size in batch_sizes:
            print(f"Benchmarking: workers={num_workers}, batch_size={batch_size}")
            throughput = benchmark_multi(
                tokenized_data, batch_size, num_workers, model_params['model_type'],
                model_params['model_id'], model_params['file_name'],
                args.device, args.gpu, args.threads)
            results[(num_workers, batch_size)] = throughput
            print(f"  -> {throughput:.2f} messages/s")

    # Print summary
    multi_worker = len(worker_counts) > 1 or worker_counts[0] != 1

    if args.device == 'gpu':
        gpu_name = torch.cuda.get_device_name(args.gpu)
        header = f"Dataset: {args.dataset}, Model: {args.model}, GPU: {gpu_name}"
    else:
        header = f"Dataset: {args.dataset}, Model: {args.model}, CPU threads: {args.threads}"

    print(f"\n{header}\n")

    lines = []
    if multi_worker:
        if args.device == 'cpu':
            lines.append("workers\tsize\tmessages/s\tmessages/s/thread")
            for num_workers in worker_counts:
                for batch_size in batch_sizes:
                    throughput = results[(num_workers, batch_size)]
                    lines.append(f"{num_workers}\t{batch_size}\t{throughput:.2f}\t\t{throughput/args.threads:.2f}")
        else:
            lines.append("workers\tsize\tmessages/s")
            for num_workers in worker_counts:
                for batch_size in batch_sizes:
                    lines.append(f"{num_workers}\t{batch_size}\t{results[(num_workers, batch_size)]:.2f}")
    else:
        if args.device == 'cpu':
            lines.append("size\tmessages/s\tmessages/s/thread")
            for batch_size in batch_sizes:
                throughput = results[(1, batch_size)]
                lines.append(f"{batch_size}\t{throughput:.2f}\t\t{throughput/args.threads:.2f}")
        else:
            lines.append("size\tmessages/s")
            for batch_size in batch_sizes:
                lines.append(f"{batch_size}\t{results[(1, batch_size)]:.2f}")

    for line in lines:
        print(line)

    if args.save:
        model_name = model_ids["torch"]["model_id"].split('/')[-1]
        if args.device == 'gpu':
            device_name = gpu_name.replace(' ', '-')
        else:
            device_name = f"cpu_{args.threads}t"
        save_dir = f"results/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{device_name}_{args.model}_{args.dataset}.tsv"
        with open(save_path, 'w') as f:
            for line in lines:
                f.write(line + "\n")
        print(f"\nResults saved to {save_path}")
