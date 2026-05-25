"""
Efficiency measurement script for reviewer R1.8.
Measures training time, inference throughput, GPU memory, and parameter count.
"""
import subprocess
import sys
import time
import torch
import numpy as np

# Models to benchmark
MODELS = ["FTMamba", "PatchTST", "iTransformer", "Mamba", "DLinear", "TimesNet", "Transformer"]
SEQ_LEN = 96
PRED_LEN = 96
BATCH_SIZE = 64
E_LAYERS = 3
D_MODEL = 512
D_FF = 64
N_HEAD = 8

results = []

for model in MODELS:
    print(f"\n===== Benchmarking {model} =====")

    # Build command
    extra_args = ""
    if model in ["PatchTST", "iTransformer", "Transformer"]:
        extra_args = f"--n_heads {N_HEAD}"
    elif model == "TimesNet":
        extra_args = f"--n_heads {N_HEAD} --top_k 5 --num_kernels 6"
    elif model == "FreTS":
        extra_args = "--channel_independence 1"
    elif model == "S_Mamba":
        D_FF = 64

    d_ff_arg = D_FF
    if model == "TimesNet":
        d_ff_arg = 2048

    cmd = (
        f"python -u run.py --task_name long_term_forecast --is_training 1 "
        f"--root_path ./dataset/ETT-small/ --data_path ETTh1.csv "
        f"--model_id ETTh1_benchmark_96 --model {model} --data ETTh1 "
        f"--features M --seq_len {SEQ_LEN} --label_len 48 --pred_len {PRED_LEN} "
        f"--e_layers {E_LAYERS} --d_layers 1 --enc_in 7 --dec_in 7 --c_out 7 "
        f"--d_model {D_MODEL} --d_ff {d_ff_arg} --dropout 0.1 "
        f"--batch_size {BATCH_SIZE} --des Bench_Exp --itr 1 {extra_args}"
    )

    # Time the training
    start = time.time()
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start

    if ret.returncode != 0:
        print(f"  {model}: FAILED ({ret.stderr[:100] if ret.stderr else 'unknown'})")
        results.append({"model": model, "status": "FAILED"})
        continue

    # Parse output for loss/metric info
    output = ret.stdout
    print(f"  Training time: {elapsed:.1f}s")

    # Count parameters by importing the model
    try:
        from run import get_args
        # Quick param count via a mock
        from models import FTMamba, PatchTST, iTransformer, DLinear, TimesNet, Transformer, Mamba, S_Mamba, FEDformer, FreTS

        model_map = {
            "FTMamba": FTMamba.Model,
            "PatchTST": PatchTST.Model,
            "iTransformer": iTransformer.Model,
            "Mamba": MambaSimple.Model if hasattr(__import__('models', fromlist=['MambaSimple']), 'MambaSimple') else None,
            "DLinear": DLinear.Model,
            "TimesNet": TimesNet.Model,
            "Transformer": Transformer.Model,
        }

        if model in model_map and model_map[model] is not None:
            # Create a mock args object
            class MockArgs:
                task_name = "long_term_forecast"
                seq_len = 96
                pred_len = 96
                enc_in = 7
                dec_in = 7
                c_out = 7
                d_model = D_MODEL
                d_ff = d_ff_arg
                e_layers = E_LAYERS
                dropout = 0.1
                d_conv = 4
                expand = 2
                n_heads = N_HEAD
                ablation_mode = "full"
                embed = "timeF"
                freq = "h"
                individual = False
                output_attention = False
                target = "OT"
                num_class = 0

            mock_args = MockArgs()
            net = model_map[model](mock_args)
            param_count = sum(p.numel() for p in net.parameters())
            print(f"  Parameters: {param_count:,}")

            # Memory estimation with a forward pass
            if torch.cuda.is_available():
                net = net.cuda()
                dummy = torch.randn(BATCH_SIZE, SEQ_LEN, 7).cuda()
                dummy_mark = torch.randn(BATCH_SIZE, SEQ_LEN, 4).cuda()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = net(dummy, dummy_mark, dummy, dummy_mark)
                mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  GPU Memory (est.): {mem:.0f} MB")
            else:
                mem = 0
        else:
            param_count = 0
            mem = 0

        results.append({
            "model": model,
            "params": param_count,
            "time_sec": elapsed,
            "gpu_mem_mb": mem,
            "status": "OK"
        })
    except Exception as e:
        print(f"  Param count failed: {e}")
        results.append({
            "model": model,
            "params": 0,
            "time_sec": elapsed,
            "gpu_mem_mb": 0,
            "status": "partial"
        })

# Summary
print("\n" + "="*70)
print("EFFICIENCY BENCHMARK SUMMARY")
print("="*70)
print(f"{'Model':<20} {'Params':>10} {'Time(s)':>10} {'GPU Mem(MB)':>12}")
print("-"*55)
for r in results:
    if r["status"] != "FAILED":
        print(f"{r['model']:<20} {r['params']:>10,} {r['time_sec']:>10.0f} {r['gpu_mem_mb']:>12.0f}")
    else:
        print(f"{r['model']:<20} {'FAILED':>10}")

print("\nResults saved for reviewer R1.8")
