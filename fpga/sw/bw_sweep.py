#!/usr/bin/env python3
"""Cycle cost of the Laya kernel shape vs. effective shared-DDR bandwidth (RTL simulation).

    env/bin/python fpga/sw/bw_sweep.py
bw = beats/100 cycles for read+write together; MB/s figures assume the 100 MHz target clock.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lmm_ref  # noqa: E402
import run_sim  # noqa: E402

run_sim.VLT = run_sim.FPGA / 'build/vl_bw/tb_lmm'
run_sim.build()
rng = np.random.default_rng(1)
qw = rng.integers(-32767, 32768, (2304, 768)).astype(np.int16)
rows = []
for L in (109, 557):
    qx = rng.integers(-32767, 32768, (L, 768)).astype(np.int16)
    ref = lmm_ref.contract(qx, qw)
    for bw in (0, 100, 62, 37, 25):
        y, p = run_sim.simulate(qx, qw, stall=0, lat=8, bw=bw)
        compute = -(-L // 4) * 288 * 192
        row = {'L': L, 'bw_beats_per_100': bw, 'MBps_at_100MHz': bw * 8 if bw else None,
               'cycles': p['cycles'], 'compute_cycles': compute, 'overhead': p['cycles'] / compute - 1,
               'exact': bool((y == ref).all())}
        rows.append(row)
        print(row, flush=True)
(run_sim.FPGA / 'results/bw_sweep.json').write_text(json.dumps(rows, indent=2))
