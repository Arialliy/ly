# TLFP fixed local Docker execution

Reuse image `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92` and the previously validated `route_closeout_20261012/dependencies/site` overlay. No installs or package changes. New declarative mount/task spec SHA256 is `5bfad7e42ece4bca4a5e814a05ff2d2c49b8cc5cb3285db145c910db3cd05fef`; `ENV_SPEC.json` records the values. The original project, spatial reference, dependencies, checkpoints and data mount read-only; only the new TLFP folder is writable. Numerical jobs have network disabled and CPU thread limits2, memory20GiB.

Source CPU tests and shared S/T initialization already exist. The following fresh agent invocation performs the actual TRAIN-only frozen-Plus GPU checks. It does not perform an optimizer update, a formal training run, or new development prediction. It verifies zero gate, real detection-loss gradients at zero and nonzero gates, frozen state, actual4×40×128×128 batch memory, temporal chunks, spatial phase/halo, and bypass. Autocast is diagnostic only; FP32 remains the training/evaluation protocol.

Execute this exact command ON THE HOST once:

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/tlfp_pilot_20261012/code/launch.py --kind preflight --gpu 3 --tag doc_follow_v2
```

Expected exit0 and `checks/GPU_INTEGRATION_doc_follow_v2.json` status PASS with two conditions. The launcher refuses GPU3 if >=500MiB occupied and writes a blocked receipt; do not stop external jobs, alter the invocation or retry. Report any mismatch without repair. Receipt/log: `runtime/preflight_doc_follow_v2.json` / `.log`. New tags are required for any later authorized retry; previous failures are preserved.

Formal jobs are separate, only after protocol freeze and gates pass:

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/tlfp_pilot_20261012/code/launch.py --kind train --condition S --gpu 3
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/tlfp_pilot_20261012/code/launch.py --kind train --condition T --gpu 3
```

They must be serialized if only GPU3 is free. A training label is unique and may not be overwritten. Subsequent inference uses `--kind infer --condition S|T --sequence DJI_0026_2|DJI_0067_2 --gpu 3`. T runs normal/current/bypass within the same loaded checkpoint. Scoring commands are in `../evaluation_code/README.md`. GPU1/2/3 are user-authorized; check current occupancy before every launch. Fixed seed46 and no automatic repeats/new seeds.

First preflight failed before model execution because the original loss was imported before its loader established the networks path. Its receipt remains preserved. v2 loads the original detector first, then imports its unchanged criterion. No formal updates occurred.
