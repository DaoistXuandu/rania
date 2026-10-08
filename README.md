# LMM-16: FPGA GEMM accelerator for Laya on DE10-Nano

PERURI Chip Hackathon 2026. Accelerates Laya's hottest kernel, `[L,768] x [768,2304]`
(`attn.Wqkv` + `mlp.Wi`, 44 calls per inference), on a Cyclone V (DE10-Nano).

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) at commit `a4a8921` (release 0.3.28).
Laya's source is not copied here; install it from upstream to rerun the profiling and e2e scripts.

## Layout

| Path | What | Evidence |
|---|---|---|
| `profiling_v1.py`, `profiling_v2.py`, `multilingual_profiling.py`, `baseline_multilingual.py` | find where Laya spends time | `profiling_v1/`, `profiling_v2/` (operator, module, latency CSVs; raw traces omitted for size) |
| `verification_v1.py` | check the kernel extraction and number formats against Laya decisions | `verification_v1_results/` (input tensors omitted for size) |
| `fpga/` | v1 RTL, testbench, Python harness | `fpga/results/` (W8A8, W8-only and BF16 flip near-tie decisions, so the contract is W16A16) |
| `v2/script/` | v2 RTL (DSP sum-of-2 PE, rounding folded into acc init, full scrub) | `v2/script/results/`, table in `v2/script/README.md` |
| `v2/proposal/` | proposal source and PDF | `proposal.pdf` |

## Reproduce (from `v2/script/`)

```bash
make lint sec sec-vl random fixtures syn   # minutes
make e2e-rtl                               # full Laya inference with the RTL in the loop, ~1 h
```

## Results (v2)

- 72/72 fixtures and 11/11 random tests bit-exact against `sw/lmm_ref.py`
- security suite `make sec` 32/32
- Yosys: 13,812 to 11,997 LUT vs v1, identical cycle counts

Cycle counts are simulated. Times in ms assume a 100 MHz clock, valid only after Quartus timing closure on the board.
