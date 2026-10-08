# RANIA: Routing Accelerator for Neural Inference Applications

An FPGA accelerator based on Laya, built for PERURI Chip Hackathon 2026. Its GEMM core, LMM-16
(the `lmm_*` files), accelerates Laya's hottest kernel, `[L,768] x [768,2304]`
(`attn.Wqkv` + `mlp.Wi`, 44 calls per inference), on a Cyclone V (DE10-Nano).

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) at commit `a4a8921` (release 0.3.28).
Laya's source is not copied here; install it from upstream to rerun the profiling and e2e scripts.
The model checkpoint is not included either; the scripts download it from Hugging Face
(`convaiinnovations/laya`, subfolder `multilingual`) on first run.

## How the work fits together

1. **Profiling** found where Laya spends its time: the `[L,768] x [768,2304]` matmul.
2. **Verification** extracted that kernel and tested number formats against Laya's decisions.
   W8A8, W8-only and BF16 flip near-tie decisions, so the contract is W16A16:
   `Y[M,N] = round((X[M,K] . W[N,K]^T) / 2^10)` as int32.
3. **`fpga/` (v1)** is the first accelerator, with RTL, testbench and harness.
4. **`v2/script/`** is the improved accelerator. Every output word is bit-identical to v1, with fewer LUTs and a full scrub.
5. **`v2/proposal/`** is the final proposal built from those results.

## Files

### Profiling and verification (repo root)

| File | Function |
|---|---|
| `baseline_multilingual.py` | loads the Laya multilingual checkpoint, prints dtype/parameter size and a baseline run |
| `multilingual_profiling.py` | quick first profile: parameter tensor sizes and a `torch.profiler` pass |
| `profiling_v1.py` | CPU/FP32 profiling on synthetic length cases (64-512 tokens); writes `profiling_v1/` |
| `profiling_v2.py` | same profiling over 12 authored scenarios with per-module scopes; writes `profiling_v2/` |
| `profiling_v1/`, `profiling_v2/` | `*_operators.csv`, `*_modules.csv`, `*_layer_operations.csv`, `*_latency.csv`, `*_summary.json`, input cases (raw traces omitted for size) |
| `verification_v1.py` | extracts the hot kernel from layers 0/11/21, checks it and exports FP32 fixtures |
| `verification_v1_results/` | kernel checks, randomized latency, verification summary, fixture manifest (input tensors omitted for size) |

### FPGA RTL (`fpga/rtl/` = v1, `v2/script/rtl/` = v2, same file roles)

| File | Function |
|---|---|
| `lmm_top.v` | top level. CSR on an Avalon-MM slave (behind the HPS lightweight bridge). Fail-closed request validator: no bus transaction unless every check passes. Window registers are write-once until reset; argument registers are frozen while busy, so there is no time-of-check/time-of-use gap. Also has the watchdog, perf counters and IRQ. |
| `lmm_core.v` | compute core. Scheduler and loader, a 4x8 PE array (dot4 int16, 128 MAC/cycle), round+shift and drain, ping-pong buffers for W panels and outputs, and a scrub that zeroes every buffer after each request. |
| `lmm_rd.v` | Avalon-MM burst read master (F2S SDRAM port 0, read-only) for X and W. Each burst is checked against the allowed window before it is issued (second guard layer). |
| `lmm_wr.v` | Avalon-MM burst write master (F2S port 1) for Y. Same per-burst guard, plus a fence read so DONE is never raised before the data lands in DDR. |
| `lmm_ram.v` | simple dual-port RAM with registered read (infers M10K). Read enable gates the output register to save power. |

v2 changes, all in `lmm_core.v`:
- The dot4 is split into two registered sum-of-2 halves, one Cyclone V DSP each.
- The rounding constant is folded into the accumulator init.
- The scrub also clears the M10K output registers.

Details and evidence are in `v2/script/README.md`.

### Testbench (`tb/`)

| File | Function |
|---|---|
| `tb_lmm.v` | Avalon DDR model with random stalls, latency and shared bandwidth. Always-on bus monitor: every access must stay inside the request's X/W/Y region, and there is no access while idle. `+test=func` runs one GEMM; `+test=sec` runs the negative/security suite. |
| `lmm.gtkw` (v1) | GTKWave view for `make wave` |

### Host software (`sw/`)

| File | Function |
|---|---|
| `lmm_ref.py` | integer reference of the contract. The RTL must match it bit-exactly. |
| `run_sim.py` | drives the Icarus/Verilator simulation and compares Y bit-exactly with `lmm_ref` (random shapes and Laya fixtures) |
| `rtl_kernel.py` | Laya kernel adapter backed by the RTL simulation: quantize, run the RTL, dequantize, record cycles |
| `e2e.py` | decision gate. Replaces the 44 `Wqkv`/`Wi` modules with a candidate kernel; the candidate must reproduce the baseline decision. |
| `bw_sweep.py` | kernel cycle cost vs effective shared-DDR bandwidth |
| `vcd2svg.py` | renders VCD signals into an SVG timeline for the report |
| `bench_platforms.py` (v2) | measured CPU vs GPU (M1, MPS) latency for the kernel and for a full Laya decision |
| `bench_table.py` (v2) | `results/bench_platforms.json` -> `../proposal/bench_table.tex` |

### Results (`results/`)

| File | Content |
|---|---|
| `sim_random.json`, `sim_fixtures.json` | bit-exact results (random shapes, 72 Laya tensors) and error vs FP32 |
| `e2e_*.json`, `log_*.txt` (v1) | decision gate per number format (w8, w12a8, w16a16, bf16, fp16, ...) and with the RTL in the loop (`e2e_rtl.json`) |
| `bw_sweep.json` (v1) | cycles vs bandwidth |
| `wave.svg`, `wave_zoom.svg` (v1) | simulation waveform figures |
| `bench_platforms.json` (v2) | CPU/GPU latency measurements |

### Build and reports

| Path | Function |
|---|---|
| `Makefile` | `lint`, `sec`, `sec-vl`, `random`, `fixtures`, `e2e-ref`, `e2e-rtl`, `bw`, `wave`, `syn` (Yosys estimate, not a Quartus fit), `pdf` |
| `fpga/report/` | v1 proposal source (`proposal.tex`, `proposal.html`, `figs/`) |
| `v2/proposal/` | final proposal (`proposal.tex`, `proposal.pdf`, `bench_table.tex`) |
| `PROPOSAL_LMM16_DE10-Nano.pdf` | submitted proposal PDF |

## Reproduce (from `v2/script/`)

Requirements: `brew install icarus-verilog verilator yosys gtkwave`, Python with torch and laya.

```bash
make lint sec sec-vl random fixtures syn   # minutes
make e2e-rtl                               # full Laya inference with the RTL in the loop, ~1 h
```

## Results (v2)

- 72/72 fixtures and 11/11 random tests bit-exact against `sw/lmm_ref.py`
- security suite `make sec` 32/32
- Yosys: 13,812 to 11,997 LUT vs v1, identical cycle counts

Cycle counts are simulated. Times in ms assume a 100 MHz clock, valid only after Quartus timing closure on the board.
