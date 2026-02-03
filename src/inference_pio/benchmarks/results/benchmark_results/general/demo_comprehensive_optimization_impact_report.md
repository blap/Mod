# Final Comprehensive Optimization Impact Benchmark Report
Generated on: 2026-01-18 09:15:55
Total Duration: 39.34 seconds
## Executive Summary
- Average Speed Improvement: +0.64%
- Average Throughput Improvement: -0.01%
- Average Memory Usage Improvement: +0.00%
- Models with Preserved Accuracy: 4/4

## Individual Model Results
### glm_4_7
- **Inference Speed**: 174.81 → 177.61 tokens/sec (+1.60%)
- **Throughput**: 9.92 → 9.92 reqs/sec (+0.01%)
- **Memory Usage**: 0.00 → 0.00 MB (+0.00%)
- **Accuracy Preserved**: Yes (Score: 1.000)
  - Unoptimized: Total time 0.51s, Avg time 0.1030s per inference
  - Optimized: Total time 0.51s, Avg time 0.1013s per inference
  - Memory: Unoptimized avg peak 0.00MB, Optimized avg peak 0.00MB

### qwen3_4b_instruct_2507
- **Inference Speed**: 175.61 → 177.46 tokens/sec (+1.05%)
- **Throughput**: 9.92 → 9.92 reqs/sec (+0.01%)
- **Memory Usage**: 0.00 → 0.00 MB (+0.00%)
- **Accuracy Preserved**: Yes (Score: 1.000)
  - Unoptimized: Total time 0.51s, Avg time 0.1025s per inference
  - Optimized: Total time 0.51s, Avg time 0.1014s per inference
  - Memory: Unoptimized avg peak 0.00MB, Optimized avg peak 0.00MB

### qwen3_coder_30b
- **Inference Speed**: 177.63 → 177.96 tokens/sec (+0.19%)
- **Throughput**: 9.91 → 9.93 reqs/sec (+0.19%)
- **Memory Usage**: 0.00 → 0.00 MB (+0.00%)
- **Accuracy Preserved**: Yes (Score: 1.000)
  - Unoptimized: Total time 0.51s, Avg time 0.1013s per inference
  - Optimized: Total time 0.51s, Avg time 0.1011s per inference
  - Memory: Unoptimized avg peak 0.00MB, Optimized avg peak 0.00MB

### qwen3_vl_2b
- **Inference Speed**: 178.12 → 177.63 tokens/sec (-0.27%)
- **Throughput**: 9.92 → 9.90 reqs/sec (-0.27%)
- **Memory Usage**: 0.00 → 0.00 MB (+0.00%)
- **Accuracy Preserved**: Yes (Score: 1.000)
  - Unoptimized: Total time 0.51s, Avg time 0.1011s per inference
  - Optimized: Total time 0.51s, Avg time 0.1013s per inference
  - Memory: Unoptimized avg peak 0.00MB, Optimized avg peak 0.00MB

## Conclusion
The optimization techniques resulted in an average speed improvement of +0.64%.
Memory usage increased by an average of +0.00%.
Model accuracy was preserved in 100.0% of cases.
