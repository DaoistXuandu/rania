# LMM-16 accelerator: build, verify, report. Run from fpga/.
PY   ?= ../env/bin/python
RTL  := $(sort $(wildcard rtl/*.v))
TB   := tb/tb_lmm.v
CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome

.PHONY: sec sec-vl random fixtures e2e-ref e2e-rtl bw wave syn pdf all

build/tb.vvp: $(RTL) $(TB)
	mkdir -p build && iverilog -g2012 -o $@ $(TB) $(RTL)

sec: build/tb.vvp            ## security/negative suite (Icarus)
	cd build && vvp -n tb.vvp +test=sec

sec-vl:                      ## same suite under Verilator
	verilator --binary --timing -O3 -Wno-fatal -Wno-lint -Wno-style --top-module tb_lmm -Mdir build/vl -o tb_lmm $(TB) $(RTL) -j 8
	cd build && ./vl/tb_lmm +test=sec

random:                      ## shapes/tails/stalls, bit-exact vs lmm_ref (use SIM=icarus to switch)
	$(PY) sw/run_sim.py random --sim $(or $(SIM),verilator)

fixtures:                    ## 72 Laya tensors, bit-exact + error vs FP32
	$(PY) sw/run_sim.py fixtures -j 6

e2e-ref:                     ## decision gate: W16A16 contract on all 44 modules
	$(PY) sw/e2e.py --out results/e2e_contract_w16a16.json

e2e-rtl:                     ## full Laya inference with the RTL simulation in the loop (~1 h)
	$(PY) sw/e2e.py --kernel rtl --out results/e2e_rtl.json

bw:                          ## cycles vs shared DDR bandwidth
	$(PY) sw/bw_sweep.py

wave: build/tb.vvp           ## VCD for GTKWave + SVG for the report
	$(PY) -c "import sys; sys.path.insert(0,'sw'); import numpy as np, run_sim; run_sim.build('icarus'); r=np.random.default_rng(7); run_sim.simulate(r.integers(-32767,32768,(9,64)).astype(np.int16), r.integers(-32767,32768,(64,64)).astype(np.int16), vcd=True)"
	python3 sw/vcd2svg.py build/lmm.vcd results/wave_zoom.svg 1800
	@echo "gtkwave build/lmm.vcd tb/lmm.gtkw"

syn:                         ## resource estimate (Yosys Mistral flow, not a Quartus fit)
	mkdir -p build/syn && yosys -q -p "read_verilog -sv $(RTL); synth_intel_alm -family cyclonev -top lmm_top; tee -o build/syn/stat.txt stat"

pdf:                         ## proposal PDF at the repo root
	"$(CHROME)" --headless=new --no-pdf-header-footer --print-to-pdf=../PROPOSAL_LMM16_DE10-Nano.pdf "file://$(CURDIR)/report/proposal.html"

all: sec random fixtures bw syn
