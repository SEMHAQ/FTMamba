"""
Collect experiment results and format them into paper-ready tables.
Run after all experiments complete: python collect_results.py
"""

import os
import json
import glob
import re


def parse_result_file(filepath):
    """Parse a TSLib result file and extract MSE/MAE."""
    results = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        # Look for the final result line
        # Format: mse:X.XX, mae:X.XX
        matches = re.findall(r'mse[:\s]*([\d.]+).*?mae[:\s]*([\d.]+)', content, re.IGNORECASE)
        if matches:
            last = matches[-1]
            results['mse'] = float(last[0])
            results['mae'] = float(last[1])
    except Exception as e:
        pass
    return results


def collect_all_results(results_dir='results'):
    """Collect all results from the results directory."""
    all_results = {}

    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return all_results

    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith('.txt') or f.endswith('.log'):
                filepath = os.path.join(root, f)
                results = parse_result_file(filepath)
                if results:
                    # Parse filename: model_dataset_predlen_...txt
                    parts = f.replace('.txt', '').replace('.log', '').split('_')
                    if len(parts) >= 3:
                        model = parts[0]
                        dataset = parts[1]
                        pred_len = parts[2]
                        key = (model, dataset, pred_len)
                        all_results[key] = results

    return all_results


def format_table(all_results, datasets, models, pred_lens):
    """Format results into a LaTeX-ready table."""
    # Header
    header = f"{'Model':<20}"
    for ds in datasets:
        for pl in pred_lens:
            header += f" & {pl}"
        header += " &"
    print(header)
    print("-" * len(header) * 2)

    # Data rows
    for model in models:
        row = f"{model:<20}"
        for ds in datasets:
            for pl in pred_lens:
                key = (model, ds, pl)
                if key in all_results:
                    mse = all_results[key].get('mse', '-')
                    mae = all_results[key].get('mae', '-')
                    row += f" & {mse:.4f}/{mae:.4f}"
                else:
                    row += f" & -/-"
        print(row)


def format_latex_table(all_results, datasets, models, pred_lens):
    """Format results as LaTeX table."""
    ncols = 1 + len(datasets) * len(pred_lens)
    col_spec = "l" + "|".join(["c" * len(pred_lens)] * len(datasets))

    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Long-term forecasting results. Lower MSE and MAE indicate better performance. Best results in \\textbf{bold}.}")
    print(f"\\label{{tab:forecasting}}")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    # Header row 1: datasets
    header1 = "Model"
    for ds in datasets:
        header1 += f" & \\multicolumn{{{len(pred_lens)}}}{{c}}{{{ds}}}"
    header1 += " \\\\"
    print(header1)

    # Header row 2: pred_lens
    header2 = ""
    for _ in datasets:
        for pl in pred_lens:
            header2 += f" & {pl}"
    header2 += " \\\\"
    print("\\cmidrule(lr){2-5}" * len(datasets))
    print(header2)
    print("\\midrule")

    # Data rows
    for model in models:
        row = model
        for ds in datasets:
            for pl in pred_lens:
                key = (model, ds, pl)
                if key in all_results:
                    mse = all_results[key].get('mse', '-')
                    if isinstance(mse, float):
                        row += f" & {mse:.4f}"
                    else:
                        row += f" & -"
                else:
                    row += f" & -"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")


if __name__ == "__main__":
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'Weather']
    models = ['FTMamba', 'PatchTST', 'iTransformer', 'Mamba', 'DLinear', 'TimesNet', 'Transformer']
    pred_lens = ['96', '192', '336', '720']

    all_results = collect_all_results()

    if not all_results:
        print("No results found. Run experiments first.")
        print("Usage: bash run_all_experiments.sh")
    else:
        print("\n=== Results Summary ===\n")
        format_table(all_results, datasets, models, pred_lens)
        print("\n=== LaTeX Table ===\n")
        format_latex_table(all_results, datasets, models, pred_lens)
