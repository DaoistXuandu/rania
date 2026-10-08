"""Laya kernel adapter backed by the RTL simulation (host side of the contract).

matmul(x, w_nk): quantize on the host, run the Verilator model of lmm_top, dequantize.
Weights are quantized once per module (warm cache), activations every call. Simulator
wall-clock is NOT FPGA time; cycle counts are kept in CYCLES for the report.
"""
import numpy as np

import lmm_ref
import run_sim

_wcache = {}
CYCLES = []


def matmul(x, w_nk):
    key = (w_nk.__array_interface__['data'][0], w_nk.shape)
    if key not in _wcache:
        _wcache[key] = lmm_ref.quantize_rows(w_nk)
    qw, sw = _wcache[key]
    qx, sx = lmm_ref.quantize_rows(x)
    y32, perf = run_sim.simulate(qx, qw)
    assert perf['passed_tb'], perf
    CYCLES.append(perf['cycles'])
    return y32.astype(np.float32) * np.float32(2 ** lmm_ref.SHIFT) * sx[:, None] * sw[None, :]


if not run_sim.VLT.exists():
    run_sim.build()
