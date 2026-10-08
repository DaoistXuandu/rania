#!/usr/bin/env python3
"""Measured CPU vs GPU latency for the Laya hot kernel and for a full Laya decision.

    env/bin/python v2/script/sw/bench_platforms.py            # -> v2/script/results/bench_platforms.json

Kernel: Y[L,2304] = X[L,768] . W[768,2304] (attn.Wqkv / mlp.Wi shape), weights resident on the
device, X uploaded and Y read back every call (a decision engine gets X from the CPU tokenizer).
End-to-end: laya.predict on one authored case per length, CPU vs the integrated GPU (MPS).
Only latency is measured here; watts come from datasheets and are labelled as such in the proposal.
"""
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ['USE_TF'] = '0'
LS = (109, 173, 301, 557)


def timeit(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(t)


def kernel_rows():
    rng = np.random.default_rng(0)
    w = torch.from_numpy(rng.standard_normal((768, 2304)).astype(np.float32))
    devs = [('cpu', torch.float32)]
    if torch.backends.mps.is_available():
        devs += [('mps', torch.float32), ('mps', torch.float16)]
    rows = []
    for L in LS:
        x = torch.from_numpy(rng.standard_normal((L, 768)).astype(np.float32))
        ref = x @ w
        for dev, dt in devs:
            wd = w.to(dev, dt)

            def call():
                y = (x.to(dev, dt) @ wd).float().cpu()          # .cpu() synchronises
                return y
            rel = float((call() - ref).norm() / ref.norm())
            ms = timeit(call, 50)
            rows.append({'L': L, 'device': dev, 'dtype': str(dt).split('.')[1], 'median_ms': ms,
                         'gflops': 2 * L * 768 * 2304 / ms / 1e6, 'rel_err_vs_fp32_cpu': rel,
                         'ms_x44': 44 * ms})
            print(rows[-1], flush=True)
    return rows


def e2e_rows():
    import laya
    from verification_v1 import QUESTIONS
    summary = json.loads((ROOT / 'verification_v1_results/verification_summary.json').read_text())
    cases = {c['id']: c for c in summary['cases']}
    rows = []
    for dev in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
        agent = laya.load('convaiinnovations/laya', subfolder='multilingual', device=dev)
        agent.model.eval()
        for cid in ('billing_64', 'billing_128', 'billing_256', 'billing_512'):
            text = cases[cid]['text']

            def call():
                with torch.inference_mode():
                    r = agent.predict(text, QUESTIONS, max_len=1024)['answers']['department']
                if dev == 'mps':
                    torch.mps.synchronize()
                return r
            choice = call()['choice']
            ms = timeit(call, 20, warm=3)
            rows.append({'case': cid, 'device': dev, 'median_ms': ms, 'choice': choice})
            print(rows[-1], flush=True)
        del agent
    return rows


if __name__ == '__main__':
    torch.set_num_threads(4)
    out = {'machine': f'{platform.machine()} {platform.platform()}', 'torch': torch.__version__,
           'threads': 4, 'kernel': kernel_rows(), 'e2e': e2e_rows()}
    p = Path(__file__).resolve().parents[1] / 'results/bench_platforms.json'
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    # self-check: every device reproduces the CPU decision, every kernel stays near FP32
    ch = {}
    for r in out['e2e']:
        ch.setdefault(r['case'], set()).add(r['choice'])
    assert all(len(v) == 1 for v in ch.values()), ch
    assert all(r['rel_err_vs_fp32_cpu'] < 1e-2 for r in out['kernel'])
    print('->', p)
