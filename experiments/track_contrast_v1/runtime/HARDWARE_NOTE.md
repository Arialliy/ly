# Hardware scheduling note

Formal A/B/C training uses physical GPUs 0/1/2. Read-only NVIDIA telemetry is
recorded every 30 seconds in `GPU_TELEMETRY.jsonl`. Software thermal slowdown
was observed on GPU1, then all three GPUs. No GPU power or clock settings were
changed. Elapsed training time includes hardware throttling and any documented
pause; it is not a comparable model-efficiency measurement across conditions.

At root authorization, the owned `track_contrast_formal_B` container on GPU1
undergoes an auditable 15-second Docker pause, followed by resume and 30 seconds
of observation. `COOLING_B.jsonl` records container identity/receipt, GPU readings,
training steps and elapsed time before/after. Model source, weights, optimizer,
step count and data order remain unchanged in memory. The controller resumes its
own pause on normal completion, SIGINT or SIGTERM. No other container is managed.

The default controller performs this single trial only. Repeated scheduling
requires separate root authorization and is recorded by a `mode: loop` event;
its fixed trigger is GPU1 thermal-active and SM clock below 1000 MHz, checking
every 5 seconds, pausing 15 seconds, then allowing at least 30 seconds after
resume. A `runtime/STOP_COOLING_B` marker stops that controller.

Single-trial result (2026-10-12 16:10:00–16:10:46 UTC): before the pause,
the latest 31-step rate was 0.6994 steps/s. The 30.382 seconds after resume
completed 22 steps (0.7241 steps/s); including the pause, 23 steps completed
in 45.681 seconds (0.5035 steps/s). GPU1 briefly cooled from 84°C/210 MHz to
79°C/1695 MHz, but within five seconds reached 87°C with hardware and software
thermal slowdown. It ended the observation at 84°C/1365 MHz with software
thermal slowdown. This short trial did not show an overall throughput benefit;
the controller exited, with the container unpaused, without starting its loop.
