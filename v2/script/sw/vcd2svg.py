#!/usr/bin/env python3
"""Render selected 1-bit/bus signals of a VCD as a compact SVG timeline (report figure).

    python3 fpga/sw/vcd2svg.py fpga/build/lmm.vcd fpga/results/wave.svg [max_cycles]
Open the same VCD interactively with: gtkwave fpga/build/lmm.vcd fpga/tb/lmm.gtkw
"""
import sys

SIGNALS = [  # (label, hierarchical name)
    ('csr_write', 'tb_lmm.csr_write'),
    ('core.st', 'tb_lmm.dut.u_core.st'),
    ('rd_read', 'tb_lmm.rd_read'),
    ('rd_valid', 'tb_lmm.rd_readdatavalid'),
    ('mac (v2)', 'tb_lmm.dut.u_core.v2'),
    ('drain', 'tb_lmm.dut.u_core.dact'),
    ('wr_write', 'tb_lmm.wr_write'),
    ('fence rd', 'tb_lmm.wr_read'),
]
STATES = ['IDLE', 'LDX', 'LDW', 'PANEL', 'FLUSH', 'FENCE', 'STOP', 'SCRUB']


def parse(path, wanted):
    ids, scope, t, changes = {}, [], 0, {n: [] for n in wanted}
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == '$scope':
                scope.append(tok[2])
            elif tok[0] == '$upscope':
                scope.pop()
            elif tok[0] == '$var':
                name = '.'.join(scope + [tok[4]])
                if name in wanted:
                    ids.setdefault(tok[3], []).append(name)
            elif tok[0][0] == '#':
                t = int(tok[0][1:])
            elif tok[0][0] in '01xz' and tok[0][1:] in ids:
                for n in ids[tok[0][1:]]:
                    changes[n].append((t, tok[0][0]))
            elif tok[0][0] == 'b' and len(tok) == 2 and tok[1] in ids:
                for n in ids[tok[1]]:
                    changes[n].append((t, tok[0][1:]))
    return changes, t


def main(vcd, out, cycles=None):
    changes, tend = parse(vcd, {n for _, n in SIGNALS})
    if cycles:
        tend = int(cycles) * 10000
    W, H, L, row = 900, 22, 80, 26
    sx = lambda t: L + (W - L - 10) * t / tend
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{row * len(SIGNALS) + 30}" '
           f'font-family="monospace" font-size="11">']
    for r, (label, name) in enumerate(SIGNALS):
        y0 = 10 + r * row
        svg.append(f'<text x="2" y="{y0 + 14}">{label}</text>')
        ch = [c for c in changes[name] if c[0] < tend] + [(tend, None)]
        for (t0, v), (t1, _) in zip(ch, ch[1:]):
            x0, x1 = sx(t0), sx(t1)
            if name.endswith('.st'):
                s = STATES[int(v, 2)] if all(c in '01' for c in v) else '?'
                svg.append(f'<rect x="{x0:.1f}" y="{y0}" width="{max(x1 - x0, .5):.1f}" height="{H - 4}" '
                           f'fill="#dde7f5" stroke="#3b5b8c" stroke-width=".5"/>')
                if x1 - x0 > 28:
                    svg.append(f'<text x="{x0 + 2:.1f}" y="{y0 + 13}" font-size="9">{s}</text>')
            elif v == '1':
                svg.append(f'<rect x="{x0:.1f}" y="{y0 + 2}" width="{max(x1 - x0, .4):.1f}" height="{H - 8}" fill="#2f6f4f"/>')
        svg.append(f'<line x1="{L}" y1="{y0 + H - 4}" x2="{W - 10}" y2="{y0 + H - 4}" stroke="#bbb" stroke-width=".5"/>')
    ycap = 10 + len(SIGNALS) * row + 12
    svg.append(f'<text x="{L}" y="{ycap}">0</text><text x="{W - 90}" y="{ycap}">{tend // 10000} cycles</text></svg>')
    open(out, 'w').write('\n'.join(svg))


if __name__ == '__main__':
    main(*sys.argv[1:4])
