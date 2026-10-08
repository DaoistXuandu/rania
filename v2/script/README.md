# LMM-16 v2: RTL, testbench, and harness

v2 of `fpga/` (which is kept unchanged as v1). Same contract, interface, register map and cycle
counts; every Y word is bit-identical to v1. The proposal built from these results is
[`../proposal/proposal.pdf`](../proposal/proposal.pdf) (source `../proposal/proposal.tex`).

## What changed from v1

| Where | Change | Why | Evidence |
|---|---|---|---|
| `rtl/lmm_core.v` PE array | dot4 split into two registered sum-of-2 halves (`ph0`, `ph1`); rounding `2^9` folded into the accumulator's first k-word; drain is `acc[41:10]` | one Cyclone V DSP per half in 18x18 sum mode with its output register (stops multiplier glitches, cuts the M10K to DSP to fabric path); 32 rounding adders (42-bit) removed | Yosys: 13,812 to 11,997 LUT; 72/72 fixtures and 11/11 random bit-exact, identical cycles |
| `rtl/lmm_core.v` scrub | last scrub cycle re-reads word 0 of every RAM | v1 left the last X/W/Y words in the M10K output registers after scrub | `tb` now checks every `q`, `ph0/ph1`, `acc`; negative control (scrub_end forced to 0) fails the 3 scrub tests |
| `tb/tb_lmm.v` | scrub check extended | see above | `make sec` 32/32 |
| `sw/run_sim.py` | random suite adds M=2, K=768, all -32768 | -32768 is a legal int16 from untrusted memory; checks the 33-bit pair sums and the accumulator bound at max K | `results/sim_random.json` 11/11 |
| `sw/bench_platforms.py`, `sw/bench_table.py` | measured CPU vs GPU (M1, MPS) latency for the kernel and for a full Laya decision | business case in the proposal | `results/bench_platforms.json` |

## Run (from `v2/script/`)

```bash
make lint sec sec-vl random fixtures syn   # minutes
make e2e-rtl                               # full Laya inference with the RTL in the loop, ~1 h
make bw                                    # cycles vs shared DDR bandwidth
../../env/bin/python sw/bench_platforms.py # run on a cool, idle machine (passive laptops throttle)
python3 sw/bench_table.py && make pdf      # proposal table + PDF (tectonic)
```

Cycle counts are simulated. Times in ms assume the 100 MHz target clock, which only becomes valid
after a Quartus timing closure on the board.
