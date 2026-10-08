#!/usr/bin/env python3
"""Laya multilingual CPU/FP32 profiling.

Run in the same environment as baseline_multilingual.py:
    python -m pip install psutil
    python profiling_v1.py

Writes v1_summary.json, v1_latency.csv, v1_operators.csv,
v1_layer_operations.csv, v1_modules.csv and one Chrome trace per case into
profiling_v1/.
Synthetic length cases measure performance, not general model accuracy.
RSS peaks are sampled; the OS high-water mark is process-lifetime, not phase-local.
Profiler memory columns are allocation counters, not process peak RAM.
"""

import argparse
import csv
import json
import math
import os
import platform
import resource
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ.pop("LAYA_CPU_AMP", None)
MIB = 1024 ** 2
RESULT_PREFIX = "v1"

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this message?",
        "criteria": {
            "billing": "payments, invoices, duplicate charges and refunds",
            "technical": "application crashes, bugs and system errors",
            "other": "anything unrelated to billing or technical problems",
        },
    }
}
BASE_CASES = [
    ("billing", "Saya ditagih dua kali. Tolong kembalikan pembayaran yang duplikat."),
    ("technical", "Aplikasi selalu crash ketika saya membuka pengaturan."),
]


def percentile(values, q):
    """Nearest-rank percentile; works for any configured repeat count."""
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def high_water_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / MIB if sys.platform == "darwin" else value / 1024


class RamSampler:
    def __init__(self, process, interval):
        self.process = process
        self.interval = interval
        self.done = threading.Event()

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def loop(self):
        while not self.done.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.before = self.process.memory_info().rss
        self.peak = self.before
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()
        self.sample()
        self.after = self.process.memory_info().rss

    def report(self):
        return {
            "rss_before_mib": self.before / MIB,
            "rss_after_mib": self.after / MIB,
            "sampled_peak_rss_mib": self.peak / MIB,
            "process_lifetime_peak_rss_mib": high_water_mib(),
            "sampling_interval_ms": self.interval * 1000,
        }


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_path(out, name):
    return out / f"{RESULT_PREFIX}_{name}"


def tensor_shapes(value, torch):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): tensor_shapes(v, torch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_shapes(v, torch) for v in value]
    return None


def make_cases(tokenizer, targets):
    cases = []
    filler = " Ini merupakan informasi tambahan untuk melengkapi laporan pengguna."
    for expected, base in BASE_CASES:
        cases.append({"id": expected + "_original", "text": base,
                      "expected": expected, "target_state_tokens": None})
        for target in targets:
            text = base
            ids = tokenizer.encode(text, add_special_tokens=False)
            if target < len(ids):
                raise ValueError(f"Target {target} terlalu pendek untuk mempertahankan pesan utama")
            while len(ids) < target:
                text += filler
                ids = tokenizer.encode(text, add_special_tokens=False)
            text = tokenizer.decode(ids[:target], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
            cases.append({"id": f"{expected}_{target}", "text": text,
                          "expected": expected, "target_state_tokens": target})
    # Decode/encode boundaries can change counts: always report actual counts.
    for case in cases:
        case["actual_state_tokens"] = len(tokenizer.encode(
            case["text"], add_special_tokens=False))
    return cases


def install_module_scopes(model, record_function):
    """Instrument all modules so functional operations retain their parent layer."""
    handles = []
    stacks = defaultdict(list)

    def pre(label):
        def hook(module, args):
            scope = record_function(label)
            scope.__enter__()
            stacks[id(module)].append(scope)
        return hook

    def post(module, args, output):
        stack = stacks[id(module)]
        if stack:
            stack.pop().__exit__(None, None, None)

    try:
        for name, module in model.named_modules():
            label = "MODULE::" + (name or "<model>")
            handles.append(module.register_forward_pre_hook(pre(label)))
            handles.append(module.register_forward_hook(post, always_call=True))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def operation_rows(prof, case_id, repeats):
    events = [e for e in prof.events() if e.name.startswith("aten::")]
    total = sum(e.self_cpu_time_total for e in events)
    groups = defaultdict(lambda: [0.0, 0, 0])
    for event in events:
        parent = event.cpu_parent
        layer = "<outside model scopes>"
        while parent is not None:
            if parent.name.startswith("MODULE::"):
                layer = parent.name.removeprefix("MODULE::")
                break
            parent = parent.cpu_parent
        shape = json.dumps(event.input_shapes)
        key = (layer, event.name, shape)
        group = groups[key]
        group[0] += event.self_cpu_time_total
        group[1] += 1
        group[2] += event.self_cpu_memory_usage
    rows = []
    for (layer, op, shape), (us, count, memory) in groups.items():
        rows.append({"case": case_id, "module": layer, "operation": op,
                     "input_shapes": shape, "calls": count,
                     "calls_per_inference": count / repeats,
                     "self_cpu_ms": us / 1000,
                     "self_cpu_ms_per_inference": us / 1000 / repeats,
                     "share_of_aten_self_cpu_pct": us / total * 100 if total else 0,
                     "net_self_tensor_allocation_bytes": memory})
    return sorted(rows, key=lambda x: x["self_cpu_ms"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="profiling_v1")
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-ms", type=float, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()
    if min(args.repeats, args.profile_repeats, args.warmup, args.threads,
           args.max_len, *args.lengths) < 1 or args.sample_ms <= 0:
        parser.error("Semua jumlah, panjang, dan interval harus positif")
    if max(args.lengths) + 256 > args.max_len:
        parser.error("Sisakan 256 token untuk pertanyaan: lengths + 256 <= max-len")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import psutil
    except ImportError:
        parser.exit(1, "Install dahulu: python -m pip install psutil\n")
    process = psutil.Process()
    interval = args.sample_ms / 1000
    report = {"config": vars(args), "platform": platform.platform(),
              "python": sys.version, "memory": {}, "cases": [],
              "notes": [
                  "Synthetic length cases are performance probes, not accuracy evaluation.",
                  "Target length refers to state text; actual packed model shapes are recorded.",
                  "Latency runs have no profiler, module scopes or RAM sampling thread.",
                  "Memory runs are separate; sampled peaks may miss short spikes.",
                  "OS peak RSS is process-lifetime, not phase-local; RSS is not parameter size.",
                  "Operator percentages use summed aten self CPU, not predict wall time.",
                  "Profiler tensor allocation counters are not peak resident memory.",
                  "First inference excludes download/load; load may include cached download checks.",
              ]}
    print("Mengimpor runtime...", flush=True)
    with RamSampler(process, interval) as ram:
        import torch
        import laya
        from torch.profiler import profile, ProfilerActivity, record_function
    report["memory"]["runtime_import"] = ram.report()
    torch.set_num_threads(args.threads)
    report["versions"] = {"torch": torch.__version__,
                          "laya": getattr(laya, "__version__", "unknown"),
                          "psutil": psutil.__version__}
    print("Memuat multilingual CPU FP32...", flush=True)
    with RamSampler(process, interval) as ram:
        start = time.perf_counter()
        agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
        load_s = time.perf_counter() - start
    agent.model.eval()
    report["memory"]["model_load"] = ram.report()
    parameters = list(agent.model.parameters())
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in parameters):
        raise RuntimeError("Baseline harus seluruhnya CPU FP32")
    report["model"] = {"load_seconds": load_s, "device": str(agent.device),
                       "parameter_count": sum(p.numel() for p in parameters),
                       "parameter_mib": sum(p.numel() * p.element_size() for p in parameters) / MIB,
                       "buffer_mib": sum(b.numel() * b.element_size() for b in agent.model.buffers()) / MIB,
                       "cpu_threads": torch.get_num_threads()}
    def predict(text):
        return agent.predict(text, QUESTIONS, max_len=args.max_len)

    # This is truly the first predict in this process, before case generation/warm-up.
    print("Mengukur inferensi pertama...", flush=True)
    with RamSampler(process, interval) as ram:
        start = time.perf_counter()
        first = predict(BASE_CASES[0][1])
        first_ms = (time.perf_counter() - start) * 1000
    report["memory"]["first_inference"] = ram.report()
    report["first_inference"] = {"wall_ms_with_ram_sampler": first_ms,
                                  "prediction": first["answers"]["department"]["choice"]}
    cases = make_cases(agent.tok, args.lengths)
    result_path(out, "input_cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mengukur {len(cases)} pesan; v1_input_cases.json menyimpan teks aktual.",
          flush=True)
    latency_rows, all_ops, all_layers = [], [], []
    modules = []
    for name, module in agent.model.named_modules():
        modules.append({"name": name or "<model>", "class": type(module).__name__,
                        "direct_parameter_shapes": json.dumps({
                            n: list(p.shape) for n, p in module.named_parameters(recurse=False)})})
    write_csv(result_path(out, "modules.csv"), modules)

    for case in cases:
        text = case["text"]
        print(f"\n{case['id']}: state={case['actual_state_tokens']} token", flush=True)
        for _ in range(args.warmup):
            predict(text)
        times, correct = [], 0
        for i in range(args.repeats):
            start = time.perf_counter()
            result = predict(text)
            ms = (time.perf_counter() - start) * 1000
            prediction = result["answers"]["department"]["choice"]
            correct += int(prediction == case["expected"])
            times.append(ms)
            latency_rows.append({"case": case["id"], "iteration": i + 1,
                                 "wall_ms": ms, "prediction": prediction,
                                 "expected": case["expected"]})
        stats = {"median_ms": statistics.median(times), "p95_ms": percentile(times, .95),
                 "min_ms": min(times), "max_ms": max(times),
                 "correct_repeated_predictions": correct, "repeats": args.repeats}
        with RamSampler(process, interval) as ram:
            for _ in range(3):
                predict(text)
        case["warm_memory"] = ram.report()
        case["latency"] = stats
        print(f"Median {stats['median_ms']:.2f} ms | P95 {stats['p95_ms']:.2f} ms | "
              f"PASS {correct}/{args.repeats} | RSS peak {ram.peak / MIB:.1f} MiB", flush=True)
        report["cases"].append(case)
        write_csv(result_path(out, "latency.csv"), latency_rows)
        result_path(out, "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Complete all memory/latency measurements before any profiler allocation.
    for case in cases:
        text = case["text"]
        print(f"\nProfiling layer: {case['id']}", flush=True)
        for _ in range(args.warmup):
            predict(text)
        captured = []
        def capture(module, inputs, kwargs):
            captured.append({"args": tensor_shapes(inputs, torch),
                             "kwargs": tensor_shapes(kwargs, torch)})
        handle = agent.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            inspected = predict(text)
        finally:
            handle.remove()
        case["model_forward_inputs"] = captured
        case["usage"] = inspected.get("usage")

        handles = install_module_scopes(agent.model, record_function)
        try:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True,
                         profile_memory=True) as prof:
                for _ in range(args.profile_repeats):
                    with record_function("PREDICT::" + case["id"]):
                        predict(text)
        finally:
            for handle in handles:
                handle.remove()
        prof.export_chrome_trace(str(result_path(out, case["id"] + "_trace.json")))
        layer_rows = operation_rows(prof, case["id"], args.profile_repeats)
        all_layers.extend(layer_rows)
        aggregates = defaultdict(lambda: [0.0, 0, 0.0])
        for row in layer_rows:
            group = aggregates[(row["operation"], row["input_shapes"])]
            group[0] += row["self_cpu_ms"]
            group[1] += row["calls"]
            group[2] += row["share_of_aten_self_cpu_pct"]
        ops = [{"case": case["id"], "operation": op, "input_shapes": shape,
                "calls": value[1], "calls_per_inference": value[1] / args.profile_repeats,
                "self_cpu_ms": value[0],
                "self_cpu_ms_per_inference": value[0] / args.profile_repeats,
                "share_of_aten_self_cpu_pct": value[2]}
               for (op, shape), value in aggregates.items()]
        ops.sort(key=lambda x: x["self_cpu_ms"], reverse=True)
        all_ops.extend(ops)
        case["top_operations"] = ops[:10]
        case["top_layer_operations"] = layer_rows[:10]
        print("Operasi dominan (persen dari aten self CPU dalam profiler):")
        for row in ops[:5]:
            print(f"  {row['operation']:45} {row['share_of_aten_self_cpu_pct']:6.2f}% "
                  f"{row['input_shapes']}")
        print("Layer + operasi dominan:")
        for row in layer_rows[:3]:
            print(f"  {row['module']} | {row['operation']} | "
                  f"{row['share_of_aten_self_cpu_pct']:.2f}%")
        # Save each completed case so long runs retain useful progress.
        write_csv(result_path(out, "latency.csv"), latency_rows)
        write_csv(result_path(out, "operators.csv"), all_ops)
        write_csv(result_path(out, "layer_operations.csv"), all_layers)
        result_path(out, "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        # Release large profiler event lists before the next case's memory run.
        del prof
    print(f"\nSelesai. Hasil: {out.resolve()}")
    print(f"Parameter FP32: {report['model']['parameter_mib']:.1f} MiB")
    for phase, memory in report["memory"].items():
        print(f"{phase}: sampled RSS peak {memory['sampled_peak_rss_mib']:.1f} MiB; "
              f"OS lifetime peak {memory['process_lifetime_peak_rss_mib']:.1f} MiB")


if __name__ == "__main__":
    main()
