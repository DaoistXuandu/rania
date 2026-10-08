import os
os.environ["USE_TF"] = "0"
os.environ.pop("LAYA_CPU_AMP", None)

import time
import laya

print("Memuat checkpoint multilingual...")
start = time.perf_counter()

agent = laya.load(
    "convaiinnovations/laya",
    subfolder="multilingual",
    device="cpu",
)

print(f"Waktu load: {time.perf_counter() - start:.2f} detik")
print("Device:", agent.device)
print("Dtype:", next(agent.model.parameters()).dtype)

questions = {
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

cases = [
    (
        "Saya ditagih dua kali. Tolong kembalikan pembayaran yang duplikat.",
        "billing",
    ),
    (
        "Aplikasi selalu crash ketika saya membuka pengaturan.",
        "technical",
    ),
]

# Warm-up: tidak dimasukkan ke pengukuran berikutnya.
agent.predict(cases[0][0], questions)

correct = 0
for text, expected in cases:
    start = time.perf_counter()
    result = agent.predict(text, questions)
    elapsed_ms = (time.perf_counter() - start) * 1000

    predicted = result["answers"]["department"]["choice"]
    passed = predicted == expected
    correct += int(passed)

    print("\nInput:", text)
    print("Prediksi:", predicted)
    print("Target:", expected)
    print("Hasil:", "PASS" if passed else "FAIL")
    print(f"Waktu inferensi: {elapsed_ms:.2f} ms")
    print("Detail:", result["answers"]["department"])

print(f"\nKeputusan benar: {correct}/{len(cases)}")


import statistics
import threading
import psutil
import torch
from torch.profiler import profile, ProfilerActivity

process = psutil.Process()
mib = 1024 ** 2

# Ukuran tensor parameter, bukan total RAM proses.
parameter_bytes = sum(
    p.numel() * p.element_size()
    for p in agent.model.parameters()
)
print(f"\nMemori parameter: {parameter_bytes / mib:.1f} MiB")
print("CPU threads:", torch.get_num_threads())

print("\n=== WAKTU DAN RAM ===")

for text, expected in cases:
    # Warm-up untuk masing-masing input.
    for _ in range(5):
        agent.predict(text, questions)

    rss_before = process.memory_info().rss
    peak = [rss_before]
    stop = threading.Event()

    def sample_ram():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.wait(0.005)

    sampler = threading.Thread(target=sample_ram, daemon=True)
    sampler.start()

    times = []
    correct = 0
    try:
        for _ in range(30):
            start = time.perf_counter()
            result = agent.predict(text, questions)
            times.append((time.perf_counter() - start) * 1000)

            prediction = result["answers"]["department"]["choice"]
            correct += int(prediction == expected)

        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()

    ordered = sorted(times)
    print(f"\nInput: {text}")
    print(f"Median: {statistics.median(times):.2f} ms")
    print(f"P95: {ordered[28]:.2f} ms")
    print(f"RAM sebelum pengukuran: {rss_before / mib:.1f} MiB")
    print(f"Peak RAM tersampel: {peak[0] / mib:.1f} MiB")
    print(f"Keputusan benar: {correct}/30")

# Profiling terpisah karena profiler menambah overhead.
print("\n=== OPERASI PYTORCH ===")

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for text, _ in cases:
        agent.predict(text, questions)

print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="self_cpu_time_total",
    row_limit=20,
))
