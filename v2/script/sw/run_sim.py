#!/usr/bin/env python3
"""Drive the Icarus simulation of lmm_top and compare Y bit-exactly with lmm_ref.contract.

    env/bin/python v2/script/sw/run_sim.py random [--sim icarus]   # tail/stall/shape sweep, small GEMMs
    env/bin/python v2/script/sw/run_sim.py fixtures [-j 6]         # replay the 72 Laya fixtures (layers 0, 11, 21)
"""
import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FPGA = HERE.parent
ROOT = FPGA.parents[1]                      # repo root (v2/script -> laya)
sys.path.insert(0, str(HERE))
import lmm_ref  # noqa: E402

VVP = FPGA / 'build/tb.vvp'
VLT = FPGA / 'build/vl/tb_lmm'
SIM = 'verilator'           # same testbench; Icarus is ~200x slower, fine for the small suites


def build(sim='verilator'):
    global SIM
    SIM = sim
    srcs = [str(FPGA / 'tb/tb_lmm.v'), *map(str, sorted((FPGA / 'rtl').glob('*.v')))]
    FPGA.joinpath('build').mkdir(exist_ok=True)
    if sim == 'icarus':
        subprocess.run(['iverilog', '-g2012', '-o', str(VVP), *srcs], check=True)
    else:
        subprocess.run(['verilator', '--binary', '--timing', '-O3', '-Wno-fatal', '-Wno-lint', '-Wno-style',
                        '--top-module', 'tb_lmm', '-Mdir', str(VLT.parent), '-o', VLT.name, '-j', '8', *srcs],
                       check=True, capture_output=True)


def simulate(qx, qw, seed=1, stall=20, lat=8, bw=0, vcd=False):
    """Run one GEMM on the RTL. Returns (y_int32 [M,N], perf dict)."""
    M, K = qx.shape
    N = qw.shape[0]
    align = lambda v: (v + 4095) // 4096 * 4096
    X = 4096
    W = align(X + M * K * 2)
    Y = align(W + N * K * 2)
    image = np.zeros(Y, np.uint8)
    image[X:X + M * K * 2] = qx.astype('<i2').view(np.uint8).ravel()
    image[W:W + N * K * 2] = qw.astype('<i2').view(np.uint8).ravel()
    with tempfile.TemporaryDirectory() as d:
        mem, out = Path(d, 'mem.hex'), Path(d, 'y.hex')
        lmm_ref.write_hex64(mem, image)
        cmd = [*(['vvp', '-n', str(VVP)] if SIM == 'icarus' else [str(VLT)]), '+test=func', f'+mem={mem}', f'+out={out}', f'+M={M}', f'+K={K}',
               f'+N={N}', f'+X={X}', f'+W={W}', f'+Y={Y}', f'+seed={seed}', f'+stall={stall}', f'+lat={lat}', f'+bw={bw}']
        if vcd:
            cmd.append('+vcd')
        log = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=FPGA / 'build').stdout
        y = lmm_ref.read_hex64(out, M * N * 4, '<i4').reshape(M, N)
    perf = {k: int(v) for line in log.splitlines() if line.startswith(('PERF', 'RESULT'))
            for k, v in (t.split('=') for t in line.split()[1:]) if v.isdigit()}
    perf['passed_tb'] = 'TEST PASSED' in log
    return y, perf


def check(qx, qw, **kw):
    y, perf = simulate(qx, qw, **kw)
    ref = lmm_ref.contract(qx, qw)
    perf['mismatches'] = int((y != ref).sum())
    perf['ok'] = perf['passed_tb'] and perf['mismatches'] == 0
    return perf


def random_suite():
    rng = np.random.default_rng(2026)
    rows = []
    shapes = [(1, 64, 32), (3, 64, 64), (4, 96, 32), (5, 128, 96), (9, 64, 64), (31, 256, 64),
              (127, 64, 32), (128, 64, 64), (129, 64, 32), (130, 128, 64), (2, 768, 32)]
    for i, (M, K, N) in enumerate(shapes):
        qx = rng.integers(-32767, 32768, (M, K)).astype(np.int16)
        qw = rng.integers(-32767, 32768, (N, K)).astype(np.int16)
        if i == 0:                                   # worst-case magnitude, both signs
            qx[:] = 32767
            qw[:16] = 32767
            qw[16:] = -32767
        if (M, K, N) == (2, 768, 32):                # untrusted data: -32768 is legal int16, K at max
            qx[:] = -32768
            qw[:] = -32768
            qx[1, ::2] = 32767
        stall, lat = [(0, 1), (20, 8), (60, 30)][i % 3]
        r = check(qx, qw, seed=i + 1, stall=stall, lat=lat)
        r.update(M=M, K=K, N=N, stall=stall, lat=lat)
        rows.append(r)
        print(f"M={M:4d} K={K:4d} N={N:4d} stall={stall:2d}% lat={lat:2d}  "
              f"mismatches={r['mismatches']}  cycles={r.get('cycles')}  {'PASS' if r['ok'] else 'FAIL'}", flush=True)
    return rows


def fixture_suite(jobs, limit=None):
    fx = ROOT / 'verification_v1_results/fixtures'
    import torch
    manifest = json.loads((fx / 'manifest.json').read_text())[:limit]
    weights = {}

    def one(item):
        if item['weight_file'] not in weights:
            weights[item['weight_file']] = torch.load(fx / item['weight_file'], weights_only=True).numpy()
        w_nk = np.ascontiguousarray(weights[item['weight_file']].T)  # fixtures store [K,N]
        d = torch.load(fx / item['fixture_file'], weights_only=True)
        x, y_fp32 = d['x'].numpy(), d['y'].numpy()
        qx, sx = lmm_ref.quantize_rows(x)
        qw, sw = lmm_ref.quantize_rows(w_nk)
        y32, perf = simulate(qx, qw)
        exact = int((y32 != lmm_ref.contract(qx, qw)).sum())
        y = y32.astype(np.float32) * np.float32(2 ** lmm_ref.SHIFT) * sx[:, None] * sw[None, :]
        rel = float(np.linalg.norm(y - y_fp32) / np.linalg.norm(y_fp32))
        cos = float((y * y_fp32).sum() / np.linalg.norm(y) / np.linalg.norm(y_fp32))
        row = {'case': item['case'], 'module': item['module'], 'M': x.shape[0], 'mismatches': exact,
               'rel_fro_err_vs_fp32': rel, 'cosine_vs_fp32': cos,
               'max_abs_err_vs_fp32': float(np.abs(y - y_fp32).max()), **perf}
        print(f"{item['case']:14s} {item['module']:28s} M={x.shape[0]:3d} mismatches={exact} "
              f"rel={rel:.2e} cycles={perf.get('cycles')}", flush=True)
        return row

    with ThreadPoolExecutor(jobs) as pool:
        return list(pool.map(one, manifest))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('suite', choices=['random', 'fixtures'])
    ap.add_argument('-j', type=int, default=6)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--sim', choices=['verilator', 'icarus'], default='verilator')
    args = ap.parse_args()
    build(args.sim)
    rows = random_suite() if args.suite == 'random' else fixture_suite(args.j, args.limit)
    out = FPGA / f'results/sim_{args.suite}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    bad = [r for r in rows if r.get('mismatches') or not r.get('passed_tb')]
    print(f"{len(rows) - len(bad)}/{len(rows)} PASS -> {out}")
    sys.exit(1 if bad else 0)
