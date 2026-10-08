# LMM-16: tiled INT16 GEMM accelerator for Laya on DE10-Nano

`Y[M,N] = round((X[M,K] · W[N,K]ᵀ) / 2¹⁰)` as int32. This replaces the dominant Laya kernel
(`encoder.layers.*.attn.Wqkv` and `mlp.Wi`: `[L,768]·[768,2304]`, called 44× per inference).
The proposal is [`../PROPOSAL_LMM16_DE10-Nano.pdf`](../PROPOSAL_LMM16_DE10-Nano.pdf); its source
is `report/proposal.html`.

| Path | What |
|---|---|
| `rtl/lmm_top.v` | CSR (Avalon slave), fail-closed validator, window lock, watchdog, perf, IRQ |
| `rtl/lmm_core.v` | scheduler, loader, 4×8 PE array (dot4 int16), drain, ping-pong buffers, scrub |
| `rtl/lmm_rd.v`, `rtl/lmm_wr.v` | Avalon burst masters (F2S ports) with per-burst guard; fence read |
| `rtl/lmm_ram.v` | dual-port RAM (M10K inference) |
| `tb/tb_lmm.v` | DDR model (stalls, latency, shared bandwidth), bus monitor, security suite |
| `sw/lmm_ref.py` | integer contract (the RTL must match it bit-exactly) |
| `sw/e2e.py` | Laya decision gate: swap the 44 modules for a kernel and compare decisions |
| `sw/run_sim.py`, `sw/rtl_kernel.py` | simulation harness and Laya adapter backed by the RTL |
| `results/` | every number quoted in the proposal |

Requirements: `brew install icarus-verilog verilator yosys gtkwave`, plus the repo's `env/`
(torch + laya). Run `make sec random fixtures bw syn`, `make e2e-rtl` (~1 h), `make wave` and `make pdf`.

These numbers are simulated cycles. Times in ms assume the 100 MHz target clock, which only
becomes valid after a Quartus timing closure on the board.
