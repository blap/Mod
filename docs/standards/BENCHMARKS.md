# Benchmarking Standards

**Reference Guide:** For instructions on running and writing benchmarks, see [Benchmarking Guide](../guides/benchmarking.md).

## 1. Directory Structure
*   Core framework scripts **MUST** be in `benchmarks/core/`.
*   Model-specific benchmarks **SHOULD** be in `src/inference_pio/models/<model>/benchmarks/`.
*   Runner scripts **MUST** be in `benchmarks/scripts/`.

## 2. Naming Conventions
*   **Files:** Must start with `benchmark_` (e.g., `benchmark_latency.py`).
*   **Functions:** Must start with `benchmark_` or `run_`.

## 3. Mandatory Metrics
All model inference benchmarks **MUST** report:
1.  **Time To First Token (TTFT)**: Latency for the first generated token.
2.  **Tokens Per Second (TPS)**: Throughput for generation.
3.  **Peak Memory Usage**: Max VRAM/RAM consumed.

## 4. Implementation Rules
*   **Hardware Context:** Every result **MUST** be tagged with the output of `HardwareAnalyzer`.
*   **Output:** Results **MUST** be saved to `benchmark_results/` in JSON format.
*   **Reproducibility:** Benchmarks **MUST** set random seeds before execution.
