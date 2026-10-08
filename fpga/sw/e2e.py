#!/usr/bin/env python3
"""Decision gate: run Laya multilingual with attn.Wqkv + mlp.Wi replaced by a candidate kernel.

    env/bin/python fpga/sw/e2e.py                           # W16A16 contract on all 44 modules
    env/bin/python fpga/sw/e2e.py --abits 8 --wbits 8       # precision sweep (INT8)
    env/bin/python fpga/sw/e2e.py --kernel rtl --cases billing_64 --layers 0

The baseline is recomputed in the same process (unrounded probabilities) and also checked
against verification_v1_results/baseline_answers.json. Baseline mistakes are kept as-is:
the candidate must reproduce the baseline decision, not the expected label.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
os.environ['USE_TF'] = '0'
import lmm_ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel', choices=['ref', 'rtl', 'fp16', 'bf16', 'w8'], default='ref')
    ap.add_argument('--layers', default='all')
    ap.add_argument('--cases', default='all')
    ap.add_argument('--abits', type=int, default=16)
    ap.add_argument('--wbits', type=int, default=16)
    ap.add_argument('--out', default=str(ROOT / 'fpga/results/e2e_ref.json'))
    args = ap.parse_args()

    import torch
    import laya
    from verification_v1 import QUESTIONS
    torch.set_num_threads(4)
    summary = json.loads((ROOT / 'verification_v1_results/verification_summary.json').read_text())
    stored = json.loads((ROOT / 'verification_v1_results/baseline_answers.json').read_text())
    cases = [c for c in summary['cases'] if args.cases == 'all' or c['id'] in args.cases.split(',')]

    agent = laya.load('convaiinnovations/laya', subfolder='multilingual', device='cpu')
    agent.model.eval()
    layers = range(22) if args.layers == 'all' else [int(i) for i in args.layers.split(',')]
    mods = dict(agent.model.named_modules())
    names = [f'encoder.layers.{i}.{s}' for i in layers for s in ['attn.Wqkv', 'mlp.Wi']]
    rtl_kernel = None
    if args.kernel == 'rtl':
        import rtl_kernel
        kernel = lambda x, w, _: rtl_kernel.matmul(x, w)
    elif args.kernel == 'ref':
        kernel = lambda x, w, a: lmm_ref.matmul(x, w, a, args.wbits)
    elif args.kernel == 'w8':  # weights INT8, activations FP32: isolates weight error
        def kernel(x, w, _):
            q, s = lmm_ref.quantize_rows(w)
            return x @ (q.astype(np.float32) * s[:, None]).T
    else:  # half-precision operands, FP32 accumulate: a common deployment baseline
        dt = torch.float16 if args.kernel == 'fp16' else torch.bfloat16
        def kernel(x, w, _):
            r = lambda a: torch.from_numpy(a).to(dt).float()
            return (r(x) @ r(w).T).numpy()

    errs = {n: [] for n in names}
    originals = {n: mods[n].forward for n in names}

    def replacement(name):
        w = mods[name].weight.detach().float().numpy()  # [N,K] = native layout

        def forward(x):
            shape = x.shape
            x2 = x.detach().reshape(-1, shape[-1]).float().numpy()
            y = kernel(x2, w, args.abits)
            ref = x2 @ w.T
            errs[name].append(float(np.linalg.norm(y - ref) / np.linalg.norm(ref)))
            return torch.from_numpy(y).reshape(*shape[:-1], w.shape[0])
        return forward

    def predict(text, mode):
        for n in names:
            mods[n].forward = replacement(n) if mode == 'cand' else originals[n]
        t = time.perf_counter()
        with torch.inference_mode():
            r = agent.predict(text, QUESTIONS, max_len=1024)['answers']['department']
        return r, (time.perf_counter() - t) * 1e3

    rows = []
    for c in cases:
        base, _ = predict(c['text'], 'base')
        n0 = len(rtl_kernel.CYCLES) if args.kernel == 'rtl' else 0
        cand, ms = predict(c['text'], 'cand')
        cyc = sum(rtl_kernel.CYCLES[n0:]) if args.kernel == 'rtl' else None
        dp = max(abs(base['probabilities'][k] - cand['probabilities'][k]) for k in base['probabilities'])
        rows.append({'case': c['id'], 'expected': c['expected'], 'stored_choice': stored[c['id']]['choice'],
                     'base_choice': base['choice'], 'cand_choice': cand['choice'],
                     'same_choice': base['choice'] == cand['choice'] == stored[c['id']]['choice'],
                     'base_p': base['probabilities'], 'cand_p': cand['probabilities'],
                     'max_abs_dp': dp, 'cand_wall_ms': ms, 'rtl_cycles': cyc,
                     'rtl_calls': len(rtl_kernel.CYCLES) - n0 if args.kernel == 'rtl' else None})
        print(f"{c['id']:14s} base={base['choice']:9s} cand={cand['choice']:9s} max|dp|={dp:.4f}", flush=True)
    for n in names:
        mods[n].forward = originals[n]
    flat = [e for v in errs.values() for e in v]
    report = {'kernel': args.kernel, 'modules': len(names), 'cases': rows,
              'all_choices_preserved': all(r['same_choice'] for r in rows),
              'max_abs_dp': max(r['max_abs_dp'] for r in rows),
              'tensor_rel_fro_err': {'median': float(np.median(flat)), 'max': float(np.max(flat))},
              'per_module_max_rel_err': {n: max(v) for n, v in errs.items() if v}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['all_choices_preserved', 'max_abs_dp', 'tensor_rel_fro_err']}))
    return 0 if report['all_choices_preserved'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
