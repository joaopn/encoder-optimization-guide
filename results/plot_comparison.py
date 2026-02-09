import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_tsv(results_dir, model, gpu, variant, dataset):
    """Find the TSV file matching the given parameters. Matches gpu substring."""
    model_dir = os.path.join(results_dir, model)
    if not os.path.isdir(model_dir):
        print(f"Error: model directory not found: {model_dir}")
        sys.exit(1)

    suffix = f"_{variant}_{dataset}.tsv"
    matches = [f for f in os.listdir(model_dir) if f.endswith(suffix) and gpu in f]

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"Warning: multiple matches for gpu='{gpu}', variant='{variant}': {matches}. Using first.")
    return os.path.join(model_dir, matches[0])


def read_max_throughput(tsv_path):
    """Read a TSV and return (max_messages_per_s, batch_size_at_max, workers_at_max)."""
    df = pd.read_csv(tsv_path, sep='\t')

    # Handle both 2-col (size, messages/s) and 3-col (workers, size, messages/s) formats
    msg_col = [c for c in df.columns if 'messages/s' in c][0]
    size_col = 'size'

    idx = df[msg_col].idxmax()
    workers = int(df['workers'].iloc[idx]) if 'workers' in df.columns else 1
    return df[msg_col].iloc[idx], int(df[size_col].iloc[idx]), workers


def main():
    parser = argparse.ArgumentParser(description="Plot torch vs onnx-fp16 comparison across GPUs")
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU name substrings (e.g. 'RTX-4090,RTX-5060-Ti')")
    parser.add_argument("--dataset", type=str, default="normal",
                        help="Dataset type (default: normal)")
    parser.add_argument("--model", type=str, default="roberta-base-go_emotions",
                        help="Model directory name (default: roberta-base-go_emotions)")
    parser.add_argument("--variants", type=str, default="torch,onnx-fp16",
                        help="Comma-separated variants to compare (default: torch,onnx-fp16)")
    parser.add_argument("--figsize", type=str, default=None,
                        help="Figure size in inches, comma-separated (e.g. '8,5')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: comparison_<dataset>.png)")
    args = parser.parse_args()

    results_dir = os.path.dirname(os.path.abspath(__file__))
    gpus = [g.strip() for g in args.gpus.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]

    if args.figsize:
        figsize = tuple(float(x) for x in args.figsize.split(','))
    else:
        figsize = (max(6, len(gpus) * 2.5), 5)

    # Collect data
    data = {}  # {(gpu, variant): (throughput, batch_size, workers)}
    for gpu in gpus:
        for variant in variants:
            tsv = find_tsv(results_dir, args.model, gpu, variant, args.dataset)
            if tsv is None:
                print(f"Warning: no file found for gpu='{gpu}', variant='{variant}', dataset='{args.dataset}'")
                continue
            throughput, bs, workers = read_max_throughput(tsv)
            data[(gpu, variant)] = (throughput, bs, workers)

    if not data:
        print("No data found. Check --gpus, --model, --dataset, --variants.")
        sys.exit(1)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(gpus))
    n_variants = len(variants)
    bar_width = 0.8 / n_variants
    colors = plt.cm.Set2(np.linspace(0, 0.5, n_variants))

    # Store bar objects per variant for arrow drawing
    bars_by_variant = {}

    legend_names = {
        'torch-base': 'torch (eager)',
        'torch': 'torch.compile',
    }

    for i, variant in enumerate(variants):
        offsets = x + (i - (n_variants - 1) / 2) * bar_width
        throughputs = []
        batch_sizes = []
        worker_counts = []
        for gpu in gpus:
            if (gpu, variant) in data:
                t, bs, w = data[(gpu, variant)]
                throughputs.append(t)
                batch_sizes.append(bs)
                worker_counts.append(w)
            else:
                throughputs.append(0)
                batch_sizes.append(0)
                worker_counts.append(0)

        label = legend_names.get(variant, variant)
        bars = ax.bar(offsets, throughputs, bar_width, label=label, color=colors[i], edgecolor='black', linewidth=0.5)
        bars_by_variant[variant] = (bars, throughputs, batch_sizes)

        for bar, bs, w in zip(bars, batch_sizes, worker_counts):
            if bs > 0:
                w_label = 'worker' if w == 1 else 'workers'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'bs={bs}\n{w} {w_label}', ha='center', va='bottom', fontsize=6)

    # Draw speedup arrows from first variant to last variant for each GPU
    if len(variants) >= 2:
        bars_a, throughputs_a, _ = bars_by_variant[variants[0]]
        bars_b, throughputs_b, _ = bars_by_variant[variants[-1]]

        for j in range(len(gpus)):
            if throughputs_a[j] > 0 and throughputs_b[j] > 0:
                speedup = throughputs_b[j] / throughputs_a[j]
                bar_a = bars_a[j]
                bar_b = bars_b[j]

                x_start = bar_a.get_x() + bar_a.get_width() / 2
                x_end = bar_b.get_x() + bar_b.get_width() / 2
                y_arrow = min(throughputs_a[j], throughputs_b[j]) * 0.5

                ax.annotate('', xy=(x_end, y_arrow), xytext=(x_start, y_arrow),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
                ax.text((x_start + x_end) / 2, y_arrow,
                        f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Clean up GPU labels for display (strip NVIDIA-GeForce- prefix)
    gpu_labels = []
    for gpu in gpus:
        label = gpu.replace('NVIDIA-GeForce-', '').replace('NVIDIA-', '').replace('-', ' ')
        gpu_labels.append(label)

    ax.set_xticks(x)
    ax.set_xticklabels(gpu_labels)
    ax.set_ylabel('messages/s')
    if args.dataset == 'filtered':
        ax.set_title(f'{args.model} (reddit comments > 200 char)')
    else:
        ax.set_title(f'{args.model} (reddit comments)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)

    plt.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(results_dir, f"comparison_{args.model}_{args.dataset}.png")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
