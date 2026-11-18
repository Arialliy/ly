# C-v2 hardware observations

The owned read-only telemetry process stopped normally via the requested stop marker. It recorded 26 samples at 30-second intervals from 2026-10-12T17:53:42.814752+00:00 to 2026-10-12T18:06:12.821568+00:00.

| Physical GPU | Observed peak temperature (C) | Observed peak memory (MiB) | Thermal-active samples |
| --- | --- | --- | --- |
| 0 | 84 | 12505 | 3 |
| 1 | 85 | 12505 | 9 |
| 2 | 87 | 12505 | 4 |

All three 256-step continuations and six development inference jobs completed. No sampled temperature reached 90 C, and the three-consecutive-sample alert was not triggered. No GPU controls or cooling pauses were applied in this C-v2 experiment.

Thermal slowdown was sampled on all three cards. Training elapsed times were CONT 239.804 s, FRAME 312.999 s, GAP 271.893 s. These times are descriptive hardware-confounded observations, not evidence of relative model efficiency. All three runs reported the same PyTorch peak allocated memory of 5,766,688,768 bytes.

The 30-second observations may miss shorter temperature/memory peaks. Physical GPU memory and SM clocks include driver allocation and host/idle phases. Exact sample data, hashes, and completed-run receipts are recorded in GPU_TELEMETRY_SUMMARY.json.
