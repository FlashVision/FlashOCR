"""
Benchmark Model
============================
Measure inference speed (FPS, latency) and model size.
"""
from flashocr.analytics import Benchmark

bench = Benchmark(
    model_path="workspace/my_model/best.pth",
    device="cuda",
    input_size=(32, 128),
    warmup_runs=50,
    benchmark_runs=200,
)

results = bench.run()

print(f"FPS:          {results['fps']:.1f}")
print(f"Latency:      {results['latency_ms']:.2f} ms")
print(f"Parameters:   {results['params']:,}")
print(f"FP16 Size:    {results['fp16_size_mb']:.2f} MB")
print(f"GFLOPs:       {results['gflops']:.3f}")
