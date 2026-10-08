#!/usr/bin/env python3
"""results/bench_platforms.json -> ../proposal/bench_table.tex (table + \\BenchEeSpeedup macro)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
d = json.loads((HERE / 'results/bench_platforms.json').read_text())
PEAK_GPU = 2.6e3 * 7 / 8          # GFLOPS FP32, M1 GPU scaled to the 7-core unit tested
fmt = lambda v, n=1: f'{v:,.{n}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

k = {(r['L'], r['device'], r['dtype']): r for r in d['kernel']}
e = {(r['case'], r['device']): r for r in d['e2e']}
rows = []
for L, case in zip((109, 173, 301, 557), ('billing_64', 'billing_128', 'billing_256', 'billing_512')):
    c, g = k[(L, 'cpu', 'float32')], k[(L, 'mps', 'float32')]
    ec, eg = e[(case, 'cpu')], e[(case, 'mps')]
    rows.append(f"{L} & {fmt(c['median_ms'], 2)} & {fmt(g['median_ms'], 2)} & "
                f"{fmt(100 * g['gflops'] / PEAK_GPU, 0)}\\% & {fmt(ec['median_ms'], 0)} & {fmt(eg['median_ms'], 0)} & "
                f"{fmt(ec['median_ms'] / eg['median_ms'], 2)}$\\times$ \\\\ \\hline")
sp = [e[(c, 'cpu')]['median_ms'] / e[(c, 'mps')]['median_ms'] for c in ('billing_64', 'billing_128', 'billing_256', 'billing_512')]
tex = r"""{\small
\begin{tabularx}{\linewidth}{|r|R|R|R|R|R|R|}
\hline
\hd{L} & \hd{Kernel CPU, ms\U} & \hd{Kernel GPU, ms\U} & \hd{Utilisasi GPU vs puncak} & \hd{Keputusan penuh CPU, ms\U} & \hd{Keputusan penuh GPU, ms\U} & \hd{GPU vs CPU} \\ \hline
""" + '\n'.join(rows) + r"""
\end{tabularx}}
\captionof{table}{Median; kernel 50 ulangan, keputusan penuh 20 ulangan; FP32, 4 thread; keputusan GPU = CPU di semua kasus (\code{v2/script/results/bench\_platforms.json}). Mesin laptop pendingin pasif; ulangi setelah cooldown karena throttling menggeser angka CPU.}
"""
tex += '\\newcommand{\\BenchEeSpeedup}{' + f'{fmt(min(sp), 1)}--{fmt(max(sp), 1)}$\\times$' + '}\n'
out = HERE.parent / 'proposal/bench_table.tex'
out.write_text(tex)
assert len(rows) == 4 and all(s > 0 for s in sp)
print(tex)
